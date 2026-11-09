import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { BrainHemisphere, BrainMapAssets, BrainSurfaceKind } from '../types/brainMap';
import type { CanonicalRoi } from '../types/session';
import type { SelectivityAtlasPoint, SelectivityBrainHoverDatum } from '../types/selectivityAtlas';
import { fetchVerifiedBrainAsset } from '../services/brainMapAssets';
import { BRAIN_SURFACE_VIEWS, brainSurfaceCameraDistance, inflatedHemisphereTranslations, type BrainSurfaceView } from '../utils/brainSurfaceLayout';
import { SELECTIVITY_CLUSTER_COLORS } from '../utils/selectivityAtlas';
import { projectSelectivityHemisphere } from '../utils/selectivityBrainProjection';
import { projectVoxelClusters } from '../utils/projectedVoxelClusters';
import { buildCategoricalSurfaceGeometry } from '../utils/categoricalSurfaceOverlay';
import { selectivityNavigationIndex } from '../utils/selectivityViewerLayout';

type SelectivityBrainViewerProps = {
  assets: BrainMapAssets;
  points: SelectivityAtlasPoint[];
  enabledRois: ReadonlySet<CanonicalRoi>;
  surface: BrainSurfaceKind;
  cameraView?: BrainSurfaceView;
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

type OverlayHitData = {
  overlay: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
  sourceVertexIndices: Uint32Array;
  anchorsByVertex: Map<number, SourceAnchor>;
};

type Runtime = {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  surfaceRoot: THREE.Group;
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

/** Surface labels are winners of mapped native-voxel support, never averaged cluster IDs. */
export default function SelectivityBrainViewer({
  assets,
  points,
  enabledRois,
  surface,
  cameraView = 'ventral',
  selectedCluster,
  resetKey,
  onHover,
  onSelectCluster,
}: SelectivityBrainViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<Runtime | null>(null);
  const cameraViewRef = useRef(cameraView);
  const onHoverRef = useRef(onHover);
  const onSelectClusterRef = useRef(onSelectCluster);
  const hitDataRef = useRef<OverlayHitData[]>([]);
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
    // Like Cortex's 3D brain response viewer, let the page show through.
    scene.background = null;
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 3000);
    camera.up.fromArray(BRAIN_SURFACE_VIEWS[cameraViewRef.current].up);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      setLoadState('error');
      setLoadError(error instanceof Error ? `Brain renderer unavailable: ${error.message}` : 'Brain renderer unavailable: WebGL could not be initialized.');
      return;
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = 'brain-map-webgl-canvas selectivity-brain-webgl-canvas';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-describedby', 'selectivity-atlas-keyboard-instructions');
    renderer.domElement.setAttribute(
      'aria-label',
      'Interactive Subject 1 cortical surface with projected voxel clusters. Drag or use arrow keys to rotate; scroll or plus and minus to zoom; brackets or Page Up and Page Down to browse mapped voxels; Home and End for first and last; Enter or Space to select a cluster.',
    );
    mountElement.appendChild(renderer.domElement);

    let controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 35;
    controls.maxDistance = 850;
    // The pale ground and warm underside light keep ventral cortex softly lit.
    // Settings match assets/js/components/UploadBrainViewer.jsx in Cortex.
    scene.add(new THREE.HemisphereLight(0xffffff, 0xe4dfd6, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(1, 1.4, 1.8);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xa8b7dc, 1.2);
    fillLight.position.set(-1.5, 0.2, -1);
    scene.add(fillLight);
    const undersideLight = new THREE.DirectionalLight(0xfff6ee, 2.2);
    undersideLight.position.set(0.2, -1.6, 0.45);
    scene.add(undersideLight);

    const surfaceRoot = new THREE.Group();
    scene.add(surfaceRoot);
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
      const preset = BRAIN_SURFACE_VIEWS[cameraViewRef.current];
      const nextUp = new THREE.Vector3().fromArray(preset.up);
      const upChanged = !camera.up.equals(nextUp);
      camera.up.copy(nextUp);
      camera.position.copy(center).add(new THREE.Vector3().fromArray(preset.direction).normalize().multiplyScalar(distance));
      if (upChanged) {
        // OrbitControls caches its up-to-Y quaternion at construction. Recreate
        // it when changing basis so native dragging follows the selected view.
        controls.removeEventListener('change', handleViewChange);
        controls.dispose();
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.addEventListener('change', handleViewChange);
        if (runtimeRef.current) runtimeRef.current.controls = controls;
      }
      controls.minDistance = Math.max(5, framingRadius * 0.25);
      controls.maxDistance = Math.max(850, distance * 2);
      controls.target.copy(center);
      controls.update();
    }

    runtimeRef.current = { renderer, camera, controls, surfaceRoot, frameSurface, clearHover };

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
          if (child instanceof THREE.Mesh && !child.userData.selectivityClusterOverlay) corticalMeshes.push(child);
        });
        occlusionCache = null;
      }
      anchorWorldPosition(anchor, occlusionTarget);
      const coordinates = [camera.position.x, camera.position.y, camera.position.z, occlusionTarget.x, occlusionTarget.y, occlusionTarget.z];
      if (occlusionCache?.coordinates.every((coordinate, index) => coordinate === coordinates[index])) return occlusionCache.occluded;
      surfaceRoot.updateMatrixWorld(true);
      occlusionDirection.subVectors(occlusionTarget, camera.position);
      const distance = occlusionDirection.length();
      // Only geometry BEFORE the supported surface vertex can hide it. A tiny
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
      const hits = raycaster.intersectObjects(hitData.map((entry) => entry.overlay), false);
      if (hits.length === 0) return null;
      for (const hit of hits) {
        if (hit.faceIndex === undefined || hit.faceIndex === null) continue;
        const data = hitData.find((entry) => entry.overlay === hit.object);
        const vertexIndex = data?.sourceVertexIndices[hit.faceIndex];
        const anchor = vertexIndex === undefined ? undefined : data?.anchorsByVertex.get(vertexIndex);
        if (!anchor) continue;
        // Inspect any visible projected cell, using a real contributor from its
        // winning cluster. The gray cortex still occludes cells on a back bank.
        occlusionRaycaster.setFromCamera(pointer, camera);
        occlusionRaycaster.far = Math.max(0, hit.distance - 0.08);
        if (occlusionRaycaster.intersectObjects(hemisphereBaseMeshes(), false).length === 0) return anchor;
      }
      return null;
    }

    function hemisphereBaseMeshes(): THREE.Mesh[] {
      const meshes: THREE.Mesh[] = [];
      surfaceRoot.traverse((child) => {
        if (child instanceof THREE.Mesh && !child.userData.selectivityClusterOverlay) meshes.push(child);
      });
      return meshes;
    }

    function handlePointerMove(event: PointerEvent) {
      if (event.buttons !== 0) {
        if (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4) pointerDragged = true;
        clearHover();
        return;
      }
      const anchor = hitAt(event);
      if (!anchor) { clearHover(); return; }
      focusedIndex = visibleAnchorsRef.current.findIndex((source) => source.point.voxelIndex === anchor.point.voxelIndex);
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
        const offset = camera.position.clone().sub(controls.target);
        const horizontal = event.key === 'ArrowLeft' || event.key === 'ArrowRight';
        const axis = horizontal ? camera.up.clone().normalize() : offset.clone().cross(camera.up).normalize();
        const sign = event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1;
        offset.applyAxisAngle(axis, sign * step);
        camera.position.copy(controls.target).add(offset);
        controls.update();
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        camera.position.lerp(controls.target, 0.1);
        controls.update();
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        camera.position.sub(controls.target).multiplyScalar(1.1).add(controls.target);
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
    const labelCounts: Record<number, number> = {};
    for (const { hemisphere, mesh } of hemisphereMeshes) {
      mesh.children.filter((child) => child.userData.selectivityClusterOverlay).forEach((child) => {
        mesh.remove(child);
        disposeObject(child);
      });
      const position = mesh.geometry.getAttribute('position');
      const projected = projectSelectivityHemisphere({ assets, points, enabledRois, hemisphere, vertexCount: position.count });
      const anchors: SourceAnchor[] = projected.voxelAnchors.map((anchor) => ({ ...anchor, hemisphere, mesh }));
      allAnchors.push(...anchors);
      const clustered = projectVoxelClusters({ assets, points, enabledRois, hemisphere, vertexCount: position.count });
      const index = mesh.geometry.getIndex();
      if (!index) throw new Error('Projected voxel clusters require the original indexed cortical mesh.');
      const cells = buildCategoricalSurfaceGeometry({
        positions: position.array,
        normals: mesh.geometry.getAttribute('normal').array,
        indices: index.array,
        labels: clustered.labels,
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(cells.positions, 3));
      if (cells.normals) geometry.setAttribute('normal', new THREE.BufferAttribute(cells.normals, 3));
      geometry.setAttribute('clusterLabel', new THREE.BufferAttribute(cells.clusterLabels, 1));
      geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cells.positions.length), 3));
      const material = new THREE.MeshStandardMaterial({
        color: '#ffffff',
        vertexColors: true,
        roughness: 0.94,
        metalness: 0,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -1,
      });
      const overlay = new THREE.Mesh(geometry, material);
      overlay.name = `${hemisphere}-projected-voxel-clusters`;
      overlay.renderOrder = 3;
      overlay.userData.selectivityClusterOverlay = true;
      const anchorsByVertex = new Map<number, SourceAnchor>();
      clustered.contributors.forEach((contributors, vertexIndex) => {
        const label = clustered.labels[vertexIndex];
        // Contributors are sorted by descending weight, then stable source ID.
        const winner = contributors.find((contributor) => contributor.point.clusterLabel === label);
        if (!winner) return;
        labelCounts[label] = (labelCounts[label] ?? 0) + 1;
        anchorsByVertex.set(vertexIndex, {
          hemisphere, mesh, point: winner.point, vertexIndex,
          mappingWeight: winner.weight, sourceCount: contributors.length,
        });
      });
      mesh.add(overlay);
      hitDataRef.current.push({ overlay, sourceVertexIndices: cells.sourceVertexIndices, anchorsByVertex });
    }
    visibleAnchorsRef.current = allAnchors.sort((a, b) => a.point.voxelIndex - b.point.voxelIndex);
    runtime.renderer.domElement.dataset.visiblePointCount = String(allAnchors.length);
    runtime.renderer.domElement.dataset.projectedClusterVertexCounts = JSON.stringify(labelCounts);
  }, [assets, enabledRois, hemisphereMeshes, points]);

  useEffect(() => {
    runtimeRef.current?.clearHover();
    const baseColor = new THREE.Color(CORTEX_COLOR);
    for (const { overlay } of hitDataRef.current) {
      const labels = overlay.geometry.getAttribute('clusterLabel');
      const colors = overlay.geometry.getAttribute('color');
      const palette = SELECTIVITY_CLUSTER_COLORS.map((hex, cluster) => {
        const color = new THREE.Color(hex);
        if (selectedCluster !== null && cluster !== selectedCluster) color.lerp(baseColor, 0.7);
        return color;
      });
      for (let i = 0; i < labels.count; i += 1) {
        const color = palette[labels.getX(i)];
        colors.setXYZ(i, color.r, color.g, color.b);
      }
      colors.needsUpdate = true;
    }
  }, [enabledRois, hemisphereMeshes, points, selectedCluster]);

  useEffect(() => {
    if (resetKey > 0) runtimeRef.current?.frameSurface();
  }, [resetKey]);

  useEffect(() => {
    cameraViewRef.current = cameraView;
    runtimeRef.current?.frameSurface();
  }, [cameraView]);

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
