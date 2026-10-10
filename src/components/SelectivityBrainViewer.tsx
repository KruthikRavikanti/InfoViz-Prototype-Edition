import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { BrainHemisphere, BrainMapAssets, BrainSurfaceKind } from '../types/brainMap';
import type { CanonicalRoi } from '../types/session';
import type { SelectivityAtlasPoint, SelectivityBrainHoverDatum } from '../types/selectivityAtlas';
import { fetchVerifiedBrainAsset } from '../services/brainMapAssets';
import { brainSurfaceCameraDistance, inflatedHemisphereTranslations } from '../utils/brainSurfaceLayout';
import { SELECTIVITY_CLUSTER_COLORS } from '../utils/selectivityAtlas';
import { projectSelectivityHemisphere } from '../utils/selectivityBrainProjection';
import { orbitSelectivityCamera, selectivityNavigationIndex } from '../utils/selectivityViewerLayout';

type SelectivityBrainViewerProps = {
  assets: BrainMapAssets;
  points: SelectivityAtlasPoint[];
  enabledRois: ReadonlySet<CanonicalRoi>;
  surface: BrainSurfaceKind;
  selectedCluster: number | null;
  resetKey: number;
  onHover: (datum: SelectivityBrainHoverDatum | null) => void;
  onSelectCluster: (cluster: number) => void;
};

type HemisphereMesh = {
  hemisphere: BrainHemisphere;
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
};

type SourceAnchor = {
  hemisphere: BrainHemisphere;
  mesh: HemisphereMesh['mesh'];
  point: SelectivityAtlasPoint;
  vertexIndex: number;
  mappingWeight: number;
  sourceCount: number;
};

type PointHitData = {
  points: THREE.Points;
  anchors: SourceAnchor[];
};

type Runtime = {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  surfaceRoot: THREE.Group;
  pointTexture: THREE.CanvasTexture;
  frameSurface: () => void;
  clearHover: () => void;
};

const HEMISPHERES: BrainHemisphere[] = ['lh', 'rh'];
const CORTEX_COLOR = '#d8d3ca';

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

/** Cluster labels belong to native voxels, never to interpolated surface values. */
export default function SelectivityBrainViewer({
  assets,
  points,
  enabledRois,
  surface,
  selectedCluster,
  resetKey,
  onHover,
  onSelectCluster,
}: SelectivityBrainViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<Runtime | null>(null);
  const onHoverRef = useRef(onHover);
  const onSelectClusterRef = useRef(onSelectCluster);
  const hitDataRef = useRef<PointHitData[]>([]);
  const visibleAnchorsRef = useRef<SourceAnchor[]>([]);
  const [hemisphereMeshes, setHemisphereMeshes] = useState<HemisphereMesh[]>([]);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [loadError, setLoadError] = useState('');

  useEffect(() => { onHoverRef.current = onHover; }, [onHover]);
  useEffect(() => { onSelectClusterRef.current = onSelectCluster; }, [onSelectCluster]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const mountElement = mount;
    const scene = new THREE.Scene();
    const backgroundColor = getComputedStyle(mountElement).getPropertyValue('--bg-subtle').trim() || '#F1EFE9';
    scene.background = new THREE.Color(backgroundColor);
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 3000);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch (error) {
      setLoadState('error');
      setLoadError(error instanceof Error ? `Brain renderer unavailable: ${error.message}` : 'Brain renderer unavailable: WebGL could not be initialized.');
      return;
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = 'brain-map-webgl-canvas selectivity-brain-webgl-canvas';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-describedby', 'selectivity-atlas-keyboard-instructions');
    renderer.domElement.setAttribute(
      'aria-label',
      'Interactive Subject 1 cortical surface with voxel-cluster markers. Drag or use arrow keys to rotate; scroll or plus and minus to zoom; brackets or Page Up and Page Down to browse mapped voxels; Home and End for first and last; Enter or Space to select a cluster.',
    );
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
    const textureCanvas = document.createElement('canvas');
    textureCanvas.width = 32;
    textureCanvas.height = 32;
    const textureContext = textureCanvas.getContext('2d');
    if (textureContext) {
      textureContext.fillStyle = '#ffffff';
      textureContext.beginPath();
      textureContext.arc(16, 16, 14, 0, Math.PI * 2);
      textureContext.fill();
    }
    const pointTexture = new THREE.CanvasTexture(textureCanvas);
    const focusCanvas = document.createElement('canvas');
    focusCanvas.width = 64;
    focusCanvas.height = 64;
    const focusContext = focusCanvas.getContext('2d');
    if (focusContext) {
      focusContext.beginPath();
      focusContext.arc(32, 32, 24, 0, Math.PI * 2);
      focusContext.strokeStyle = '#ffffff';
      focusContext.lineWidth = 10;
      focusContext.stroke();
      focusContext.strokeStyle = '#292524';
      focusContext.lineWidth = 5;
      focusContext.stroke();
    }
    const focusTexture = new THREE.CanvasTexture(focusCanvas);
    const focusMarker = new THREE.Points(
      new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3)),
      new THREE.PointsMaterial({
        color: '#ffffff',
        size: 18,
        sizeAttenuation: false,
        map: focusTexture,
        alphaTest: 0.3,
        depthWrite: false,
      }),
    );
    focusMarker.visible = false;
    focusMarker.frustumCulled = false;
    focusMarker.renderOrder = 4;
    scene.add(focusMarker);

    let framingRadius = 0;
    let framingDistance = 0;
    let focusedIndex: number | null = null;
    let hoverMode: 'pointer' | 'keyboard' | null = null;
    let pointerStart: { x: number; y: number } | null = null;
    let pointerDragged = false;
    const projectedPoint = new THREE.Vector3();
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 2.5 };
    const occlusionRaycaster = new THREE.Raycaster();
    const occlusionTarget = new THREE.Vector3();
    const occlusionDirection = new THREE.Vector3();
    let cachedSurface: THREE.Object3D | undefined;
    let corticalMeshes: THREE.Mesh[] = [];
    let occlusionCache: { coordinates: number[]; occluded: boolean } | null = null;
    const pointer = new THREE.Vector2();

    function clearHover() {
      focusedIndex = null;
      hoverMode = null;
      focusMarker.visible = false;
      delete renderer.domElement.dataset.focusedVoxelIndex;
      delete renderer.domElement.dataset.focusedVoxelOccluded;
      onHoverRef.current(null);
      renderer.domElement.style.cursor = 'grab';
    }

    function flushOrbitDamping() {
      const enabled = controls.enableDamping;
      controls.enableDamping = false;
      controls.update();
      controls.enableDamping = enabled;
    }

    function fitDistance() {
      const distance = brainSurfaceCameraDistance(framingRadius, camera.aspect, camera.fov);
      controls.maxDistance = Math.max(850, distance * 2);
      camera.far = Math.max(3000, distance + framingRadius * 4);
      camera.updateProjectionMatrix();
      return distance;
    }

    function frameSurface() {
      clearHover();
      flushOrbitDamping();
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

    runtimeRef.current = { renderer, camera, controls, surfaceRoot, pointTexture, frameSurface, clearHover };

    function anchorWorldPosition(anchor: SourceAnchor, target: THREE.Vector3) {
      const position = anchor.mesh.geometry.getAttribute('position');
      target.set(position.getX(anchor.vertexIndex), position.getY(anchor.vertexIndex), position.getZ(anchor.vertexIndex));
      anchor.mesh.updateWorldMatrix(true, false);
      return anchor.mesh.localToWorld(target);
    }

    function datumFor(anchor: SourceAnchor, canvasX: number, canvasY: number, outsideView?: boolean, occluded?: boolean): SelectivityBrainHoverDatum {
      return {
        ...anchor.point,
        hemisphere: anchor.hemisphere,
        vertexIndex: anchor.vertexIndex,
        volumeCoordinateZyx: assets.projection.rois[anchor.point.roi].volumeCoordinatesZyx[anchor.point.roiVoxelIndex],
        mappingWeight: anchor.mappingWeight,
        sourceCount: anchor.sourceCount,
        canvasX,
        canvasY,
        outsideView,
        occluded,
      };
    }

    function isOccluded(anchor: SourceAnchor): boolean {
      const currentSurface = surfaceRoot.children[0];
      if (cachedSurface !== currentSurface) {
        cachedSurface = currentSurface;
        corticalMeshes = [];
        surfaceRoot.traverse((child) => {
          if (child instanceof THREE.Mesh) corticalMeshes.push(child);
        });
        occlusionCache = null;
      }
      anchorWorldPosition(anchor, occlusionTarget);
      const coordinates = [camera.position.x, camera.position.y, camera.position.z, occlusionTarget.x, occlusionTarget.y, occlusionTarget.z];
      if (occlusionCache?.coordinates.every((coordinate, index) => coordinate === coordinates[index])) return occlusionCache.occluded;
      surfaceRoot.updateMatrixWorld(true);
      occlusionDirection.subVectors(occlusionTarget, camera.position);
      const distance = occlusionDirection.length();
      // Only geometry BEFORE the supported marker vertex can hide it. A tiny
      // display-space tolerance avoids treating the vertex's own face as a hit.
      occlusionRaycaster.set(camera.position, occlusionDirection.normalize());
      occlusionRaycaster.near = 0;
      occlusionRaycaster.far = Math.max(0, distance - 0.08);
      const occluded = occlusionRaycaster.intersectObjects(corticalMeshes, false).length > 0;
      occlusionCache = { coordinates, occluded };
      return occluded;
    }

    function showFocusMarker(anchor: SourceAnchor) {
      anchorWorldPosition(anchor, projectedPoint);
      const position = focusMarker.geometry.getAttribute('position');
      position.setXYZ(0, projectedPoint.x, projectedPoint.y, projectedPoint.z);
      position.needsUpdate = true;
      focusMarker.visible = true;
      renderer.domElement.dataset.focusedVoxelIndex = String(anchor.point.voxelIndex);
    }

    function showKeyboardHover() {
      if (hoverMode !== 'keyboard' || focusedIndex === null) return;
      const anchor = visibleAnchorsRef.current[focusedIndex];
      if (!anchor) { clearHover(); return; }
      showFocusMarker(anchor);
      camera.updateMatrixWorld();
      projectedPoint.project(camera);
      const rect = renderer.domElement.getBoundingClientRect();
      const outsideView = !Number.isFinite(projectedPoint.x) || !Number.isFinite(projectedPoint.y)
        || !Number.isFinite(projectedPoint.z) || projectedPoint.z < -1 || projectedPoint.z > 1
        || Math.abs(projectedPoint.x) > 1 || Math.abs(projectedPoint.y) > 1;
      const occluded = !outsideView && isOccluded(anchor);
      focusMarker.visible = !outsideView && !occluded;
      renderer.domElement.dataset.focusedVoxelOccluded = String(occluded);
      onHoverRef.current(datumFor(
        anchor,
        Number.isFinite(projectedPoint.x) ? (projectedPoint.x + 1) * rect.width / 2 : rect.width / 2,
        Number.isFinite(projectedPoint.y) ? (1 - projectedPoint.y) * rect.height / 2 : rect.height / 2,
        outsideView,
        occluded,
      ));
    }

    function handleViewChange() {
      if (hoverMode === 'keyboard') showKeyboardHover();
      else if (hoverMode === 'pointer') clearHover();
    }
    controls.addEventListener('change', handleViewChange);

    function hitAt(event: PointerEvent): SourceAnchor | null {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      camera.updateMatrixWorld();
      surfaceRoot.updateMatrixWorld(true);
      raycaster.setFromCamera(pointer, camera);
      const hitData = hitDataRef.current;
      const hits = raycaster.intersectObjects(hitData.map((entry) => entry.points), false);
      if (hits.length === 0) return null;
      for (const hit of hits) {
        if (hit.index === undefined) continue;
        const data = hitData.find((entry) => entry.points === hit.object);
        const anchor = data?.anchors[hit.index];
        if (!anchor) continue;
        anchorWorldPosition(anchor, projectedPoint).project(camera);
        const screenX = (projectedPoint.x + 1) * rect.width / 2;
        const screenY = (1 - projectedPoint.y) * rect.height / 2;
        // Pick the same visible source identity as the depth-tested glyph, not
        // a back-bank marker that happens to be close to the pointer ray.
        if (Math.hypot(screenX - (event.clientX - rect.left), screenY - (event.clientY - rect.top)) <= 8 && !isOccluded(anchor)) return anchor;
      }
      return null;
    }

    function handlePointerMove(event: PointerEvent) {
      if (event.buttons !== 0) {
        if (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4) pointerDragged = true;
        clearHover();
        return;
      }
      const anchor = hitAt(event);
      if (!anchor) { clearHover(); return; }
      focusedIndex = visibleAnchorsRef.current.indexOf(anchor);
      hoverMode = 'pointer';
      showFocusMarker(anchor);
      renderer.domElement.dataset.focusedVoxelOccluded = 'false';
      renderer.domElement.style.cursor = 'crosshair';
      const rect = renderer.domElement.getBoundingClientRect();
      onHoverRef.current(datumFor(anchor, event.clientX - rect.left, event.clientY - rect.top));
    }

    function handlePointerDown(event: PointerEvent) {
      pointerStart = { x: event.clientX, y: event.clientY };
      pointerDragged = false;
      clearHover();
    }

    function handleClick(event: PointerEvent) {
      if (pointerDragged || (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4)) return;
      const anchor = hitAt(event);
      if (anchor) onSelectClusterRef.current(anchor.point.clusterLabel);
    }

    function handleKeyboard(event: KeyboardEvent) {
      const navigationIndex = selectivityNavigationIndex(visibleAnchorsRef.current.length, focusedIndex, event.key);
      if (navigationIndex !== null) {
        event.preventDefault();
        focusedIndex = navigationIndex;
        hoverMode = 'keyboard';
        showKeyboardHover();
      } else if (event.key.startsWith('Arrow')) {
        event.preventDefault();
        flushOrbitDamping();
        const step = Math.PI / 36;
        const azimuth = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
        const polar = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
        camera.position.fromArray(orbitSelectivityCamera(camera.position.toArray(), controls.target.toArray(), azimuth, polar));
        controls.update();
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        camera.position.lerp(controls.target, 0.08);
        controls.update();
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        camera.position.sub(controls.target).multiplyScalar(1.08).add(controls.target);
        controls.update();
      } else if ((event.key === 'Enter' || event.key === ' ') && focusedIndex !== null) {
        event.preventDefault();
        const anchor = visibleAnchorsRef.current[focusedIndex];
        if (anchor) onSelectClusterRef.current(anchor.point.clusterLabel);
      }
    }

    renderer.domElement.addEventListener('pointerdown', handlePointerDown);
    renderer.domElement.addEventListener('pointermove', handlePointerMove);
    renderer.domElement.addEventListener('pointerleave', clearHover);
    renderer.domElement.addEventListener('click', handleClick);
    renderer.domElement.addEventListener('keydown', handleKeyboard);
    renderer.domElement.addEventListener('blur', clearHover);

    function resize() {
      const { width, height } = mountElement.getBoundingClientRect();
      renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
      camera.aspect = Math.max(width, 1) / Math.max(height, 1);
      camera.updateProjectionMatrix();
      if (framingRadius > 0 && framingDistance > 0) {
        flushOrbitDamping();
        const distance = fitDistance();
        camera.position.sub(controls.target).multiplyScalar(distance / framingDistance).add(controls.target);
        framingDistance = distance;
        controls.update();
      }
      if (hoverMode === 'keyboard') showKeyboardHover();
      else clearHover();
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
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
      renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      renderer.domElement.removeEventListener('pointerleave', clearHover);
      renderer.domElement.removeEventListener('click', handleClick);
      renderer.domElement.removeEventListener('keydown', handleKeyboard);
      renderer.domElement.removeEventListener('blur', clearHover);
      controls.removeEventListener('change', handleViewChange);
      controls.dispose();
      disposeObject(surfaceRoot);
      disposeObject(focusMarker);
      pointTexture.dispose();
      focusTexture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      hitDataRef.current = [];
      visibleAnchorsRef.current = [];
      runtimeRef.current = null;
    };
  }, [assets]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    const surfaceAsset = assets.manifest.surfaces[surface];
    if (!runtime) return;
    if (!surfaceAsset) {
      setLoadState('error');
      setLoadError(`The ${surface} cortical surface is not present in the Subject 1 asset manifest.`);
      return;
    }
    const activeRuntime = runtime;
    const activeSurfaceAsset = surfaceAsset;
    let cancelled = false;
    let frameRequest = 0;
    const abortController = new AbortController();
    setLoadState('loading');
    setLoadError('');
    setHemisphereMeshes([]);
    runtime.clearHover();
    hitDataRef.current = [];
    visibleAnchorsRef.current = [];
    [...runtime.surfaceRoot.children].forEach((child) => {
      runtime.surfaceRoot.remove(child);
      disposeObject(child);
    });

    function handleLoadFailure(error?: unknown) {
      if (cancelled) return;
      setLoadState('error');
      setLoadError(error instanceof Error ? error.message : `The ${surface} cortical surface file could not be loaded.`);
    }

    function prepareSurface(gltf: GLTF) {
      if (cancelled) { disposeObject(gltf.scene); return; }
      try {
        const hemisphereNodes = Object.fromEntries(HEMISPHERES.map((hemisphere) => {
          const nodeName = activeSurfaceAsset.hemispheres[hemisphere].node;
          const node = gltf.scene.getObjectByName(nodeName);
          if (!node) throw new Error(`Surface node “${nodeName}” is missing from ${surface}.glb.`);
          return [hemisphere, node];
        })) as Record<BrainHemisphere, THREE.Object3D>;
        const meshes = HEMISPHERES.map((hemisphere): HemisphereMesh => {
          const declared = activeSurfaceAsset.hemispheres[hemisphere];
          const sourceMesh = firstMesh(hemisphereNodes[hemisphere]);
          if (!sourceMesh) throw new Error(`Surface node “${declared.node}” is missing from ${surface}.glb.`);
          const position = sourceMesh.geometry.getAttribute('position');
          const index = sourceMesh.geometry.getIndex();
          const faceCount = (index?.count ?? position?.count ?? 0) / 3;
          if (!position || position.itemSize !== 3 || position.count !== declared.vertexCount || faceCount !== declared.faceCount) {
            throw new Error(`${hemisphere.toUpperCase()} ${surface} mesh topology does not match the Subject 1 asset manifest.`);
          }
          if (index) {
            for (let i = 0; i < index.count; i += 1) {
              if (index.getX(i) < 0 || index.getX(i) >= position.count) throw new Error('Cortical surface contains an invalid vertex index.');
            }
          }
          const originalGeometry = sourceMesh.geometry;
          sourceMesh.geometry = originalGeometry.clone();
          originalGeometry.dispose();
          if (!sourceMesh.geometry.getAttribute('normal')) sourceMesh.geometry.computeVertexNormals();
          const oldMaterials = Array.isArray(sourceMesh.material) ? sourceMesh.material : [sourceMesh.material];
          oldMaterials.forEach((material) => material.dispose());
          sourceMesh.material = new THREE.MeshStandardMaterial({
            color: CORTEX_COLOR,
            roughness: 0.94,
            metalness: 0,
            side: THREE.DoubleSide,
          });
          return { hemisphere, mesh: sourceMesh as HemisphereMesh['mesh'] };
        });
        if (surface === 'inflated') {
          gltf.scene.updateMatrixWorld(true);
          const leftBox = new THREE.Box3().setFromObject(hemisphereNodes.lh);
          const rightBox = new THREE.Box3().setFromObject(hemisphereNodes.rh);
          const translations = inflatedHemisphereTranslations(
            { min: leftBox.min.x, max: leftBox.max.x },
            { min: rightBox.min.x, max: rightBox.max.x },
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
        frameRequest = window.requestAnimationFrame(activeRuntime.frameSurface);
      } catch (error) {
        disposeObject(gltf.scene);
        handleLoadFailure(error);
      }
    }

    fetchVerifiedBrainAsset(
      activeSurfaceAsset.url,
      activeSurfaceAsset.sha256,
      `${surface[0].toUpperCase()}${surface.slice(1)} cortical surface`,
      fetch,
      abortController.signal,
    ).then((bytes) => {
      if (cancelled) return;
      new GLTFLoader().parse(bytes, new URL('.', activeSurfaceAsset.url).toString(), prepareSurface, handleLoadFailure);
    }).catch(handleLoadFailure);

    return () => {
      cancelled = true;
      abortController.abort();
      window.cancelAnimationFrame(frameRequest);
    };
  }, [assets, surface]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    runtime.clearHover();
    hitDataRef.current = [];
    const allAnchors: SourceAnchor[] = [];
    const baseColor = new THREE.Color(CORTEX_COLOR);
    for (const { hemisphere, mesh } of hemisphereMeshes) {
      mesh.children.filter((child) => child.userData.selectivityVoxelPoints).forEach((child) => {
        mesh.remove(child);
        disposeObject(child);
      });
      const position = mesh.geometry.getAttribute('position');
      const projected = projectSelectivityHemisphere({ assets, points, enabledRois, hemisphere, vertexCount: position.count });
      const anchors: SourceAnchor[] = projected.voxelAnchors.map((anchor) => ({ ...anchor, hemisphere, mesh }));
      allAnchors.push(...anchors);
      if (anchors.length === 0) continue;
      const positions = new Float32Array(anchors.length * 3);
      const colors = new Float32Array(anchors.length * 3);
      anchors.forEach((anchor, index) => {
        positions[index * 3] = position.getX(anchor.vertexIndex);
        positions[index * 3 + 1] = position.getY(anchor.vertexIndex);
        positions[index * 3 + 2] = position.getZ(anchor.vertexIndex);
        const color = new THREE.Color(SELECTIVITY_CLUSTER_COLORS[anchor.point.clusterLabel]);
        if (selectedCluster !== null && anchor.point.clusterLabel !== selectedCluster) color.lerp(baseColor, 0.7);
        color.toArray(colors, index * 3);
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const material = new THREE.PointsMaterial({
        size: 6,
        sizeAttenuation: false,
        vertexColors: true,
        map: runtime.pointTexture,
        alphaTest: 0.3,
        depthWrite: false,
      });
      const voxelPoints = new THREE.Points(geometry, material);
      voxelPoints.name = `${hemisphere}-selectivity-voxels`;
      voxelPoints.renderOrder = 3;
      voxelPoints.userData.selectivityVoxelPoints = true;
      mesh.add(voxelPoints);
      hitDataRef.current.push({ points: voxelPoints, anchors });
    }
    visibleAnchorsRef.current = allAnchors.sort((a, b) => a.point.voxelIndex - b.point.voxelIndex);
    runtime.renderer.domElement.dataset.visiblePointCount = String(allAnchors.length);
  }, [assets, enabledRois, hemisphereMeshes, points, selectedCluster]);

  useEffect(() => {
    if (resetKey > 0) runtimeRef.current?.frameSurface();
  }, [resetKey]);

  return (
    <div className="brain-map-viewer-shell selectivity-brain-viewer-shell">
      <div ref={mountRef} className="brain-map-canvas" />
      {loadState !== 'ready' && (
        <div className={`brain-map-canvas-state ${loadState === 'error' ? 'error' : ''}`} role={loadState === 'error' ? 'alert' : 'status'}>
          {loadState === 'error' ? loadError : `Loading ${surface} cortical surface…`}
        </div>
      )}
    </div>
  );
}
