import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type {
  SelectivityAtlasHoverDatum,
  SelectivityAtlasPoint,
  SelectivityAtlasRoi,
} from '../types/selectivityAtlas';
import { SELECTIVITY_CLUSTER_COLORS } from '../utils/selectivityAtlas';
import {
  SELECTIVITY_DEFAULT_CAMERA_DIRECTION,
  orbitSelectivityCamera,
  resizedSelectivityCameraDistance,
  selectivityBoundsCenter,
  selectivityFrameDistance,
  selectivityNavigationIndex,
  selectivityOrientationAxes,
  type SelectivityPosition,
} from '../utils/selectivityViewerLayout';

type VoxelSelectivityAtlasViewerProps = {
  points: SelectivityAtlasPoint[];
  enabledRois: ReadonlySet<SelectivityAtlasRoi>;
  selectedCluster: number | null;
  resetKey: number;
  onHover: (datum: SelectivityAtlasHoverDatum | null) => void;
  onSelectCluster: (cluster: number) => void;
};

type Runtime = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  background: THREE.Color;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  pointCloud: THREE.Points;
  framePoints: () => void;
  clearHover: () => void;
};

function disposePointCloud(pointCloud: THREE.Points) {
  pointCloud.geometry.dispose();
  const materials = Array.isArray(pointCloud.material) ? pointCloud.material : [pointCloud.material];
  materials.forEach((material) => material.dispose());
}

function mixColor(color: THREE.Color, background: THREE.Color, amount: number): THREE.Color {
  return color.clone().lerp(background, amount);
}

function updatePointColors(
  pointCloud: THREE.Points,
  points: SelectivityAtlasPoint[],
  selectedCluster: number | null,
  background: THREE.Color,
) {
  const colors = new Float32Array(points.length * 3);
  points.forEach((point, index) => {
    const color = new THREE.Color(SELECTIVITY_CLUSTER_COLORS[point.clusterLabel]);
    const displayColor = selectedCluster !== null && point.clusterLabel !== selectedCluster
      ? mixColor(color, background, 0.82)
      : color;
    displayColor.toArray(colors, index * 3);
  });
  pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

export default function VoxelSelectivityAtlasViewer({
  points,
  enabledRois,
  selectedCluster,
  resetKey,
  onHover,
  onSelectCluster,
}: VoxelSelectivityAtlasViewerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<Runtime | null>(null);
  const orientationRef = useRef<SVGSVGElement | null>(null);
  const fullPositions = useMemo<SelectivityPosition[]>(
    () => points.map((point) => [point.umapX, point.umapY, point.umapZ]),
    [points],
  );
  const fullPositionsRef = useRef(fullPositions);
  const visiblePointsRef = useRef<SelectivityAtlasPoint[]>([]);
  const selectedClusterRef = useRef(selectedCluster);
  const onHoverRef = useRef(onHover);
  const onSelectClusterRef = useRef(onSelectCluster);

  useEffect(() => { selectedClusterRef.current = selectedCluster; }, [selectedCluster]);
  useEffect(() => { onHoverRef.current = onHover; }, [onHover]);
  useEffect(() => { onSelectClusterRef.current = onSelectCluster; }, [onSelectCluster]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const mountElement = mount;
    const backgroundColor = getComputedStyle(mountElement).getPropertyValue('--bg-subtle').trim() || '#F1EFE9';
    const background = new THREE.Color(backgroundColor);
    const scene = new THREE.Scene();
    scene.background = background;
    const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 2000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = 'selectivity-atlas-webgl-canvas';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-describedby', 'selectivity-atlas-keyboard-instructions');
    renderer.domElement.setAttribute(
      'aria-label',
      'Interactive 3D UMAP of Subject 1 voxel selectivity. Drag or use arrow keys to rotate; scroll or plus and minus to zoom; brackets or Page Up and Page Down to browse voxels; Home and End for first and last; Enter or Space to select a cluster.',
    );
    mountElement.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2;
    controls.maxDistance = 120;

    const pointCloud = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({
        size: 6,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
      }),
    );
    scene.add(pointCloud);

    const focusTextureCanvas = document.createElement('canvas');
    focusTextureCanvas.width = 64;
    focusTextureCanvas.height = 64;
    const focusContext = focusTextureCanvas.getContext('2d');
    if (focusContext) {
      focusContext.strokeStyle = '#ffffff';
      focusContext.lineWidth = 10;
      focusContext.beginPath();
      focusContext.arc(32, 32, 24, 0, Math.PI * 2);
      focusContext.stroke();
      focusContext.strokeStyle = '#292524';
      focusContext.lineWidth = 5;
      focusContext.stroke();
    }
    const focusTexture = new THREE.CanvasTexture(focusTextureCanvas);
    const focusMarker = new THREE.Points(
      new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3)),
      new THREE.PointsMaterial({
        size: 18,
        sizeAttenuation: false,
        map: focusTexture,
        transparent: true,
        depthWrite: false,
        depthTest: false,
      }),
    );
    focusMarker.visible = false;
    focusMarker.frustumCulled = false;
    focusMarker.renderOrder = 2;
    scene.add(focusMarker);

    let focusedIndex: number | null = null;
    let hoverMode: 'pointer' | 'keyboard' | null = null;
    let pointerStart: { x: number; y: number } | null = null;
    let pointerDragged = false;
    let initiallyFramed = false;
    const projectedPoint = new THREE.Vector3();

    function clearHover() {
      focusedIndex = null;
      hoverMode = null;
      focusMarker.visible = false;
      delete renderer.domElement.dataset.focusedVoxelIndex;
      onHoverRef.current(null);
      renderer.domElement.style.cursor = 'grab';
    }

    function showFocusMarker(point: SelectivityAtlasPoint) {
      const position = focusMarker.geometry.getAttribute('position');
      position.setXYZ(0, point.umapX, point.umapY, point.umapZ);
      position.needsUpdate = true;
      focusMarker.visible = true;
      renderer.domElement.dataset.focusedVoxelIndex = String(point.voxelIndex);
    }

    function showKeyboardHover() {
      if (hoverMode !== 'keyboard' || focusedIndex === null) return;
      const point = visiblePointsRef.current[focusedIndex];
      if (!point) { clearHover(); return; }
      camera.updateMatrixWorld();
      projectedPoint.set(point.umapX, point.umapY, point.umapZ).project(camera);
      const rect = renderer.domElement.getBoundingClientRect();
      const outsideView = !Number.isFinite(projectedPoint.x) || !Number.isFinite(projectedPoint.y) || !Number.isFinite(projectedPoint.z)
        || projectedPoint.z < -1 || projectedPoint.z > 1 || Math.abs(projectedPoint.x) > 1 || Math.abs(projectedPoint.y) > 1;
      showFocusMarker(point);
      focusMarker.visible = !outsideView;
      onHoverRef.current({
        ...point,
        canvasX: Number.isFinite(projectedPoint.x) ? (projectedPoint.x + 1) * rect.width / 2 : rect.width / 2,
        canvasY: Number.isFinite(projectedPoint.y) ? (1 - projectedPoint.y) * rect.height / 2 : rect.height / 2,
        outsideView,
      });
    }

    function flushOrbitDamping() {
      const enabled = controls.enableDamping;
      controls.enableDamping = false;
      controls.update();
      controls.enableDamping = enabled;
    }

    function framePoints() {
      clearHover();
      const allPositions = fullPositionsRef.current;
      if (allPositions.length === 0) return;
      flushOrbitDamping();
      const centerCoordinates = selectivityBoundsCenter(allPositions);
      const center = new THREE.Vector3(...centerCoordinates);
      const distance = selectivityFrameDistance(allPositions, centerCoordinates, camera.aspect, camera.fov);
      const cameraOffset = new THREE.Vector3(...SELECTIVITY_DEFAULT_CAMERA_DIRECTION).normalize();
      controls.maxDistance = Math.max(120, distance * 5);
      camera.position.copy(center).add(cameraOffset.multiplyScalar(distance));
      controls.target.copy(center);
      controls.update();
      initiallyFramed = true;
    }
    runtimeRef.current = { renderer, scene, background, camera, controls, pointCloud, framePoints, clearHover };

    const orientationAxes = Array.from(orientationRef.current?.querySelectorAll<SVGGElement>('[data-umap-axis]') ?? [])
      .map((group) => ({
        group,
        line: group.querySelector('line'),
        endpoint: group.querySelector('circle'),
        label: group.querySelector('text'),
      }));
    const lastOrientation = new THREE.Quaternion(0, 0, 0, 0);

    function updateOrientation() {
      if (camera.quaternion.equals(lastOrientation)) return;
      lastOrientation.copy(camera.quaternion);
      const axes = selectivityOrientationAxes(camera.quaternion.toArray());
      orientationAxes.forEach(({ group, line, endpoint, label }, index) => {
        const [x, y, z] = axes[index];
        const endX = 74 + x * 25;
        const endY = 62 - y * 25;
        line?.setAttribute('x2', String(endX));
        line?.setAttribute('y2', String(endY));
        endpoint?.setAttribute('cx', String(endX));
        endpoint?.setAttribute('cy', String(endY));
        label?.setAttribute('x', String(74 + x * 42));
        label?.setAttribute('y', String(62 - y * 42));
        group.style.opacity = String(0.65 + 0.35 * (z + 1) / 2);
      });
    }

    function handleViewChange() {
      updateOrientation();
      if (hoverMode === 'keyboard') showKeyboardHover();
      else if (hoverMode === 'pointer') clearHover();
    }
    controls.addEventListener('change', handleViewChange);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.18 };
    const pointer = new THREE.Vector2();

    function hitAt(event: PointerEvent): { index: number; point: SelectivityAtlasPoint } | null {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(pointCloud, false)[0];
      if (hit?.index === undefined) return null;
      const point = visiblePointsRef.current[hit.index];
      return point ? { index: hit.index, point } : null;
    }

    function handlePointerMove(event: PointerEvent) {
      if (event.buttons !== 0) {
        if (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4) pointerDragged = true;
        clearHover();
        return;
      }
      const hit = hitAt(event);
      if (!hit) {
        clearHover();
        return;
      }
      focusedIndex = hit.index;
      hoverMode = 'pointer';
      showFocusMarker(hit.point);
      const rect = renderer.domElement.getBoundingClientRect();
      renderer.domElement.style.cursor = 'crosshair';
      onHoverRef.current({
        ...hit.point,
        canvasX: event.clientX - rect.left,
        canvasY: event.clientY - rect.top,
      });
    }

    function handleClick(event: PointerEvent) {
      // Orbit drag ends with a click event too; do not accidentally select a cluster.
      if (pointerDragged || (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4)) return;
      const hit = hitAt(event);
      if (hit) onSelectClusterRef.current(hit.point.clusterLabel);
    }

    function handlePointerDown(event: PointerEvent) {
      pointerStart = { x: event.clientX, y: event.clientY };
      pointerDragged = false;
      clearHover();
    }

    function handleKeyboard(event: KeyboardEvent) {
      const step = Math.PI / 36;
      const navigationIndex = selectivityNavigationIndex(visiblePointsRef.current.length, focusedIndex, event.key);
      if (navigationIndex !== null) {
        event.preventDefault();
        focusedIndex = navigationIndex;
        hoverMode = 'keyboard';
        showKeyboardHover();
      } else if (event.key.startsWith('Arrow')) {
        event.preventDefault();
        flushOrbitDamping();
        const azimuth = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
        const polar = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
        camera.position.fromArray(orbitSelectivityCamera(camera.position.toArray(), controls.target.toArray(), azimuth, polar));
        controls.update();
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        camera.position.lerp(controls.target, 0.08);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        camera.position.sub(controls.target).multiplyScalar(1.08).add(controls.target);
      } else if ((event.key === 'Enter' || event.key === ' ') && focusedIndex !== null) {
        event.preventDefault();
        const point = visiblePointsRef.current[focusedIndex];
        if (point) onSelectClusterRef.current(point.clusterLabel);
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
      const oldAspect = camera.aspect;
      const newAspect = Math.max(width, 1) / Math.max(height, 1);
      if (initiallyFramed && oldAspect !== newAspect) {
        flushOrbitDamping();
        const offset = camera.position.clone().sub(controls.target);
        const allPositions = fullPositionsRef.current;
        const center = selectivityBoundsCenter(allPositions);
        const direction = offset.toArray();
        const oldFit = selectivityFrameDistance(allPositions, center, oldAspect, camera.fov, direction);
        const newFit = selectivityFrameDistance(allPositions, center, newAspect, camera.fov, direction);
        const distance = resizedSelectivityCameraDistance(offset.length(), oldFit, newFit);
        controls.maxDistance = Math.max(120, newFit * 5, distance);
        camera.position.copy(controls.target).add(offset.setLength(distance));
      }
      renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
      camera.aspect = newAspect;
      camera.updateProjectionMatrix();
      if (!initiallyFramed) framePoints();
      else {
        controls.update();
        if (hoverMode === 'keyboard') showKeyboardHover();
        else clearHover();
      }
    }
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mountElement);
    let animationFrame = 0;
    function render() {
      controls.update();
      updateOrientation();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(render);
    }
    render();

    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
      renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      renderer.domElement.removeEventListener('pointerleave', clearHover);
      renderer.domElement.removeEventListener('click', handleClick);
      renderer.domElement.removeEventListener('keydown', handleKeyboard);
      renderer.domElement.removeEventListener('blur', clearHover);
      controls.removeEventListener('change', handleViewChange);
      controls.dispose();
      disposePointCloud(pointCloud);
      disposePointCloud(focusMarker);
      focusTexture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      runtimeRef.current = null;
      visiblePointsRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (fullPositionsRef.current === fullPositions) return;
    fullPositionsRef.current = fullPositions;
    runtimeRef.current?.framePoints();
  }, [fullPositions]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    runtime.clearHover();
    const visible = points.filter((point) => enabledRois.has(point.roi));
    visiblePointsRef.current = visible;
    const positions = new Float32Array(visible.length * 3);
    visible.forEach((point, index) => {
      positions[index * 3] = point.umapX;
      positions[index * 3 + 1] = point.umapY;
      positions[index * 3 + 2] = point.umapZ;
    });
    runtime.pointCloud.geometry.dispose();
    runtime.pointCloud.geometry = new THREE.BufferGeometry();
    runtime.pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    updatePointColors(runtime.pointCloud, visible, selectedClusterRef.current, runtime.background);
    runtime.pointCloud.geometry.computeBoundingSphere();
    runtime.renderer.domElement.dataset.visiblePointCount = String(visible.length);
  }, [enabledRois, points]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    updatePointColors(runtime.pointCloud, visiblePointsRef.current, selectedCluster, runtime.background);
    runtime.clearHover();
  }, [selectedCluster]);

  useEffect(() => {
    if (resetKey > 0) runtimeRef.current?.framePoints();
  }, [resetKey]);

  return (
    <div className="selectivity-atlas-viewer-shell" ref={mountRef}>
      <svg
        ref={orientationRef}
        className="selectivity-atlas-orientation"
        viewBox="0 0 148 124"
        role="img"
        aria-label="UMAP 1, UMAP 2, and UMAP 3 axis orientation. Arbitrary similarity-space directions, not anatomical axes."
      >
        <title>Positive UMAP axes in the current camera view; arbitrary similarity-space directions, not anatomy.</title>
        <circle className="selectivity-atlas-orientation-origin" cx="74" cy="62" r="2" />
        {[1, 2, 3].map((axis) => (
          <g key={axis} className="selectivity-atlas-orientation-axis" data-umap-axis={axis}>
            <line x1="74" y1="62" x2="74" y2="62" />
            <circle cx="74" cy="62" r="2.5" />
            <text x="74" y="62" textAnchor="middle" dominantBaseline="central">UMAP {axis}</text>
          </g>
        ))}
        <text className="selectivity-atlas-orientation-caption" x="74" y="116" textAnchor="middle">Similarity axes</text>
      </svg>
    </div>
  );
}
