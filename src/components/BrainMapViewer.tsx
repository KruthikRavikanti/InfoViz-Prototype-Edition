import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type {
  BrainHemisphere,
  BrainMapAssets,
  BrainMapHoverDatum,
  BrainSurfaceKind,
  BrainSurfaceVertexDatum,
} from '../types/brainMap';
import type { BrainMapStimulusResult, CanonicalRoi } from '../types/session';
import { divergentResponseColor, projectHemisphereResponses } from '../utils/brainProjection';
import { brainSurfaceCameraDistance, inflatedHemisphereTranslations } from '../utils/brainSurfaceLayout';
import { fetchVerifiedBrainAsset } from '../services/brainMapAssets';

type BrainMapViewerProps = {
  assets: BrainMapAssets;
  stimulusResult: BrainMapStimulusResult;
  enabledRois: ReadonlySet<CanonicalRoi>;
  surface: BrainSurfaceKind;
  maxAbsResponse: number;
  resetKey: number;
  onHover: (datum: BrainMapHoverDatum | null) => void;
};

type Runtime = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  surfaceRoot: THREE.Group;
  frameSurface: () => void;
};

type HemisphereMesh = {
  hemisphere: BrainHemisphere;
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
};

type PointHitData = {
  hemisphere: BrainHemisphere;
  vertexIndices: number[];
  vertexData: BrainSurfaceVertexDatum[];
};

const HEMISPHERES: BrainHemisphere[] = ['lh', 'rh'];

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh) && !(child instanceof THREE.Points)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => material.dispose());
  });
}

function firstMesh(object: THREE.Object3D): THREE.Mesh | null {
  if (object instanceof THREE.Mesh) return object;
  let result: THREE.Mesh | null = null;
  object.traverse((child) => {
    if (!result && child instanceof THREE.Mesh) result = child;
  });
  return result;
}

export default function BrainMapViewer({
  assets,
  stimulusResult,
  enabledRois,
  surface,
  maxAbsResponse,
  resetKey,
  onHover,
}: BrainMapViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<Runtime | null>(null);
  const onHoverRef = useRef(onHover);
  const hitDataRef = useRef(new Map<string, PointHitData>());
  const [hemisphereMeshes, setHemisphereMeshes] = useState<HemisphereMesh[]>([]);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    onHoverRef.current = onHover;
  }, [onHover]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const mountElement = mount;

    const scene = new THREE.Scene();
    const backgroundColor = getComputedStyle(mountElement).getPropertyValue('--bg-subtle').trim() || '#F1EFE9';
    scene.background = new THREE.Color(backgroundColor);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 3000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = 'brain-map-webgl-canvas';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute(
      'aria-label',
      'Interactive cortical surface. Drag to rotate, scroll to zoom, or use arrow keys to rotate.',
    );
    renderer.domElement.setAttribute('role', 'img');
    mountElement.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 35;
    controls.maxDistance = 850;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x827a70, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(1, 1.4, 1.8);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xa8b7dc, 1.2);
    fillLight.position.set(-1.5, 0.2, -1);
    scene.add(fillLight);

    const surfaceRoot = new THREE.Group();
    scene.add(surfaceRoot);
    let framingRadius = 0;
    let framingDistance = 0;
    function fitDistance() {
      const distance = brainSurfaceCameraDistance(framingRadius, camera.aspect, camera.fov);
      controls.maxDistance = Math.max(850, distance * 2);
      camera.far = Math.max(3000, distance + framingRadius * 4);
      camera.updateProjectionMatrix();
      return distance;
    }
    function frameSurface() {
      surfaceRoot.rotation.set(0, 0, 0);
      const box = new THREE.Box3().setFromObject(surfaceRoot);
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      framingRadius = box.getBoundingSphere(new THREE.Sphere()).radius;
      const distance = fitDistance();
      framingDistance = distance;
      camera.position.copy(center).add(new THREE.Vector3(0.9, 0.28, 1.35).normalize().multiplyScalar(distance));
      controls.target.copy(center);
      controls.update();
    }
    runtimeRef.current = { renderer, scene, camera, controls, surfaceRoot, frameSurface };

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 2.5 };
    const pointer = new THREE.Vector2();
    function handlePointerMove(event: PointerEvent) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const points = Array.from(hitDataRef.current.keys())
        .map((uuid) => scene.getObjectByProperty('uuid', uuid))
        .filter((object): object is THREE.Points => object instanceof THREE.Points);
      const intersection = raycaster.intersectObjects(points, false)[0];
      const hit = intersection ? hitDataRef.current.get(intersection.object.uuid) : undefined;
      const pointIndex = intersection?.index;
      if (!hit || pointIndex === undefined) {
        onHoverRef.current(null);
        renderer.domElement.style.cursor = 'grab';
        return;
      }
      const datum = hit.vertexData[pointIndex];
      const dominant = datum.dominantSource;
      const projectionRoi = assets.projection.rois[dominant.roi];
      renderer.domElement.style.cursor = 'crosshair';
      onHoverRef.current({
        hemisphere: hit.hemisphere,
        vertexIndex: hit.vertexIndices[pointIndex],
        surfaceResponse: datum.surfaceResponse,
        overlappingRois: datum.overlappingRois,
        dominantSource: dominant,
        volumeCoordinateZyx: projectionRoi.volumeCoordinatesZyx[dominant.responseIndex],
        volumeLinearIndexXyz: projectionRoi.volumeLinearIndicesXyz[dominant.responseIndex],
        canvasX: event.clientX - rect.left,
        canvasY: event.clientY - rect.top,
      });
    }
    function clearHover() {
      onHoverRef.current(null);
      renderer.domElement.style.cursor = 'grab';
    }
    function handleKeyboard(event: KeyboardEvent) {
      const rotationStep = Math.PI / 36;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        surfaceRoot.rotation.y += event.key === 'ArrowLeft' ? -rotationStep : rotationStep;
      } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        event.preventDefault();
        surfaceRoot.rotation.x += event.key === 'ArrowUp' ? -rotationStep : rotationStep;
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        camera.position.lerp(controls.target, 0.08);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        camera.position.sub(controls.target).multiplyScalar(1.08).add(controls.target);
      }
    }
    renderer.domElement.addEventListener('pointermove', handlePointerMove);
    renderer.domElement.addEventListener('pointerleave', clearHover);
    renderer.domElement.addEventListener('keydown', handleKeyboard);

    function resize() {
      const { width, height } = mountElement.getBoundingClientRect();
      renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
      camera.aspect = Math.max(width, 1) / Math.max(height, 1);
      camera.updateProjectionMatrix();
      if (framingRadius > 0 && framingDistance > 0) {
        const distance = fitDistance();
        // Preserve rotation, panning, and the user's zoom relative to the fitted view.
        camera.position.sub(controls.target).multiplyScalar(distance / framingDistance).add(controls.target);
        framingDistance = distance;
        controls.update();
      }
    }
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mountElement);

    let animationFrame = 0;
    function render() {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(render);
    }
    render();

    return () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      renderer.domElement.removeEventListener('pointerleave', clearHover);
      renderer.domElement.removeEventListener('keydown', handleKeyboard);
      controls.dispose();
      disposeObject(surfaceRoot);
      renderer.dispose();
      renderer.domElement.remove();
      hitDataRef.current.clear();
      runtimeRef.current = null;
    };
  }, [assets]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    const surfaceAsset = assets.manifest.surfaces[surface];
    if (!runtime || !surfaceAsset) {
      setLoadState('error');
      setLoadError(`The ${surface} cortical surface is not present in the Subject 1 asset manifest.`);
      return;
    }
    const activeRuntime = runtime;
    const activeSurfaceAsset = surfaceAsset;

    let cancelled = false;
    const abortController = new AbortController();
    setLoadState('loading');
    setLoadError('');
    setHemisphereMeshes([]);
    onHoverRef.current(null);
    hitDataRef.current.clear();

    const previous = [...runtime.surfaceRoot.children];
    previous.forEach((child) => {
      runtime.surfaceRoot.remove(child);
      disposeObject(child);
    });

    function handleLoadFailure(error?: unknown) {
      if (cancelled) return;
      setLoadState('error');
      setLoadError(
        error instanceof Error
          ? error.message
          : `The ${surface} cortical surface file could not be loaded.`,
      );
    }

    function prepareSurface(gltf: GLTF) {
        if (cancelled) {
          disposeObject(gltf.scene);
          return;
        }
        try {
          const hemisphereNodes = Object.fromEntries(HEMISPHERES.map((hemisphere) => {
            const nodeName = activeSurfaceAsset.hemispheres[hemisphere].node;
            const node = gltf.scene.getObjectByName(nodeName);
            if (!node) throw new Error(`Surface node “${nodeName}” is missing from ${surface}.glb.`);
            return [hemisphere, node];
          })) as Record<BrainHemisphere, THREE.Object3D>;
          const meshes = HEMISPHERES.map((hemisphere): HemisphereMesh => {
            const nodeName = activeSurfaceAsset.hemispheres[hemisphere].node;
            const sourceMesh = firstMesh(hemisphereNodes[hemisphere]);
            if (!sourceMesh) throw new Error(`Surface node “${nodeName}” is missing from ${surface}.glb.`);

            const vertexCount = sourceMesh.geometry.getAttribute('position')?.count ?? 0;
            if (vertexCount !== activeSurfaceAsset.hemispheres[hemisphere].vertexCount) {
              throw new Error(
                `${hemisphere.toUpperCase()} ${surface} mesh has ${vertexCount.toLocaleString()} vertices; `
                + `${activeSurfaceAsset.hemispheres[hemisphere].vertexCount.toLocaleString()} were declared.`,
              );
            }
            const originalGeometry = sourceMesh.geometry;
            sourceMesh.geometry = originalGeometry.clone();
            originalGeometry.dispose();
            if (!sourceMesh.geometry.getAttribute('normal')) sourceMesh.geometry.computeVertexNormals();
            const oldMaterials = Array.isArray(sourceMesh.material) ? sourceMesh.material : [sourceMesh.material];
            oldMaterials.forEach((material) => material.dispose());
            sourceMesh.material = new THREE.MeshStandardMaterial({
              color: '#ffffff',
              roughness: 0.94,
              metalness: 0,
              side: THREE.DoubleSide,
              vertexColors: true,
            });
            sourceMesh.userData.brainHemisphere = hemisphere;
            return { hemisphere, mesh: sourceMesh as HemisphereMesh['mesh'] };
          });
          if (surface === 'inflated') {
            gltf.scene.updateMatrixWorld(true);
            const initialLeftBox = new THREE.Box3().setFromObject(hemisphereNodes.lh);
            const initialRightBox = new THREE.Box3().setFromObject(hemisphereNodes.rh);
            const translations = inflatedHemisphereTranslations(
              { min: initialLeftBox.min.x, max: initialLeftBox.max.x },
              { min: initialRightBox.min.x, max: initialRightBox.max.x },
              8,
            );
            hemisphereNodes.lh.position.x += translations.lh;
            hemisphereNodes.rh.position.x += translations.rh;
            gltf.scene.updateMatrixWorld(true);
            const separatedLeftBox = new THREE.Box3().setFromObject(hemisphereNodes.lh);
            const separatedRightBox = new THREE.Box3().setFromObject(hemisphereNodes.rh);
            if (separatedRightBox.min.x - separatedLeftBox.max.x < 7.99) {
              throw new Error('Inflated hemisphere display separation could not be validated.');
            }
          }
          activeRuntime.surfaceRoot.add(gltf.scene);
          setHemisphereMeshes(meshes);
          setLoadState('ready');
          window.requestAnimationFrame(activeRuntime.frameSurface);
        } catch (error) {
          disposeObject(gltf.scene);
          setLoadState('error');
          setLoadError(error instanceof Error ? error.message : 'The cortical surface could not be prepared.');
        }
    }

    fetchVerifiedBrainAsset(
      activeSurfaceAsset.url,
      activeSurfaceAsset.sha256,
      `${surface[0].toUpperCase()}${surface.slice(1)} cortical surface`,
      fetch,
      abortController.signal,
    )
      .then((bytes) => {
        if (cancelled) return;
        new GLTFLoader().parse(
          bytes,
          new URL('.', activeSurfaceAsset.url).toString(),
          prepareSurface,
          handleLoadFailure,
        );
      })
      .catch(handleLoadFailure);

    return () => {
      cancelled = true;
      abortController.abort();
    };
  }, [assets, surface]);

  useEffect(() => {
    hitDataRef.current.clear();
    for (const { hemisphere, mesh } of hemisphereMeshes) {
      const oldPoints = mesh.children.filter((child) => child.userData.brainResponsePoints);
      oldPoints.forEach((child) => {
        mesh.remove(child);
        disposeObject(child);
      });

      const position = mesh.geometry.getAttribute('position');
      const projected = projectHemisphereResponses({
        projection: assets.projection,
        roiResponses: stimulusResult.roiResponses,
        enabledRois,
        hemisphere,
        vertexCount: position.count,
      });
      const pointPositions: number[] = [];
      const vertexIndices: number[] = [];
      const vertexData: BrainSurfaceVertexDatum[] = [];
      const surfaceColors = new Float32Array(position.count * 3);
      const baseColor = new THREE.Color('#d8d3ca');
      for (let vertexIndex = 0; vertexIndex < position.count; vertexIndex += 1) {
        baseColor.toArray(surfaceColors, vertexIndex * 3);
      }
      for (let vertexIndex = 0; vertexIndex < projected.mapped.length; vertexIndex += 1) {
        const datum = projected.vertexData[vertexIndex];
        if (!projected.mapped[vertexIndex] || !datum) continue;
        pointPositions.push(position.getX(vertexIndex), position.getY(vertexIndex), position.getZ(vertexIndex));
        const [red, green, blue] = divergentResponseColor(projected.values[vertexIndex], maxAbsResponse);
        new THREE.Color().setRGB(red, green, blue, THREE.SRGBColorSpace).toArray(surfaceColors, vertexIndex * 3);
        vertexIndices.push(vertexIndex);
        vertexData.push(datum);
      }
      mesh.geometry.setAttribute('color', new THREE.BufferAttribute(surfaceColors, 3));
      mesh.material.vertexColors = true;
      mesh.material.needsUpdate = true;
      if (pointPositions.length === 0) continue;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(pointPositions, 3));
      const material = new THREE.PointsMaterial({
        color: '#ffffff',
        size: 4,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.035,
        depthWrite: false,
      });
      const points = new THREE.Points(geometry, material);
      points.name = `${hemisphere}-brain-responses`;
      points.renderOrder = 3;
      points.userData.brainResponsePoints = true;
      mesh.add(points);
      hitDataRef.current.set(points.uuid, { hemisphere, vertexIndices, vertexData });
    }
    onHoverRef.current(null);
  }, [assets.projection, enabledRois, hemisphereMeshes, maxAbsResponse, stimulusResult]);

  useEffect(() => {
    if (resetKey > 0) runtimeRef.current?.frameSurface();
  }, [resetKey]);

  return (
    <div className="brain-map-viewer-shell">
      <div ref={mountRef} className="brain-map-canvas" />
      {loadState !== 'ready' && (
        <div className={`brain-map-canvas-state ${loadState === 'error' ? 'error' : ''}`} role="status">
          {loadState === 'error' ? loadError : `Loading ${surface} cortical surface…`}
        </div>
      )}
    </div>
  );
}
