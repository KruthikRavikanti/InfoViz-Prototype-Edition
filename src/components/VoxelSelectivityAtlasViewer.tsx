import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type {
  SelectivityAtlasHoverDatum,
  SelectivityAtlasPoint,
  SelectivityAtlasRoi,
} from '../types/selectivityAtlas';
import { SELECTIVITY_CLUSTER_COLORS } from '../utils/selectivityAtlas';

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
    renderer.domElement.setAttribute(
      'aria-label',
      'Interactive 3D UMAP of Subject 1 voxel selectivity. Drag to rotate, scroll to zoom, and hover a voxel.',
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

    function framePoints() {
      const position = pointCloud.geometry.getAttribute('position');
      if (!position || position.count === 0) return;
      pointCloud.rotation.set(0, 0, 0);
      const box = new THREE.Box3().setFromObject(pointCloud);
      const center = box.getCenter(new THREE.Vector3());
      const cameraOffset = new THREE.Vector3(1, 0.55, 1.35).normalize();
      const lookDirection = cameraOffset.clone().negate();
      const cameraRight = new THREE.Vector3().crossVectors(lookDirection, camera.up).normalize();
      const cameraUp = new THREE.Vector3().crossVectors(cameraRight, lookDirection).normalize();
      const relativePoint = new THREE.Vector3();
      const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const frameFraction = 0.78;
      let distance = 2;
      for (let index = 0; index < position.count; index += 1) {
        relativePoint.fromBufferAttribute(position, index).sub(center);
        const depthOffset = relativePoint.dot(lookDirection);
        const horizontalDistance = Math.abs(relativePoint.dot(cameraRight))
          / (tanHalfFov * Math.max(camera.aspect, 0.01) * frameFraction) - depthOffset;
        const verticalDistance = Math.abs(relativePoint.dot(cameraUp))
          / (tanHalfFov * frameFraction) - depthOffset;
        distance = Math.max(distance, horizontalDistance, verticalDistance, 0.1 - depthOffset);
      }
      camera.position.copy(center).add(cameraOffset.multiplyScalar(distance));
      controls.target.copy(center);
      controls.update();
    }
    runtimeRef.current = { renderer, scene, background, camera, controls, pointCloud, framePoints };

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.18 };
    const pointer = new THREE.Vector2();
    let hoveredIndex: number | null = null;

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
      const hit = hitAt(event);
      if (!hit) {
        hoveredIndex = null;
        renderer.domElement.style.cursor = 'grab';
        onHoverRef.current(null);
        return;
      }
      hoveredIndex = hit.index;
      const rect = renderer.domElement.getBoundingClientRect();
      renderer.domElement.style.cursor = 'crosshair';
      onHoverRef.current({
        ...hit.point,
        canvasX: event.clientX - rect.left,
        canvasY: event.clientY - rect.top,
      });
    }

    function handleClick(event: PointerEvent) {
      const hit = hitAt(event);
      if (hit) onSelectClusterRef.current(hit.point.clusterLabel);
    }

    function clearHover() {
      hoveredIndex = null;
      onHoverRef.current(null);
      renderer.domElement.style.cursor = 'grab';
    }

    function handleKeyboard(event: KeyboardEvent) {
      const step = Math.PI / 36;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        pointCloud.rotation.y += event.key === 'ArrowLeft' ? -step : step;
      } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        event.preventDefault();
        pointCloud.rotation.x += event.key === 'ArrowUp' ? -step : step;
      } else if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        camera.position.lerp(controls.target, 0.08);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        camera.position.sub(controls.target).multiplyScalar(1.08).add(controls.target);
      } else if ((event.key === 'Enter' || event.key === ' ') && hoveredIndex !== null) {
        event.preventDefault();
        const point = visiblePointsRef.current[hoveredIndex];
        if (point) onSelectClusterRef.current(point.clusterLabel);
      }
    }

    renderer.domElement.addEventListener('pointermove', handlePointerMove);
    renderer.domElement.addEventListener('pointerleave', clearHover);
    renderer.domElement.addEventListener('click', handleClick);
    renderer.domElement.addEventListener('keydown', handleKeyboard);

    function resize() {
      const { width, height } = mountElement.getBoundingClientRect();
      renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
      camera.aspect = Math.max(width, 1) / Math.max(height, 1);
      camera.updateProjectionMatrix();
      framePoints();
    }
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mountElement);
    let animationFrame = 0;
    function render() {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(render);
    }
    render();

    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      renderer.domElement.removeEventListener('pointerleave', clearHover);
      renderer.domElement.removeEventListener('click', handleClick);
      renderer.domElement.removeEventListener('keydown', handleKeyboard);
      controls.dispose();
      disposePointCloud(pointCloud);
      renderer.dispose();
      renderer.domElement.remove();
      runtimeRef.current = null;
      visiblePointsRef.current = [];
    };
  }, []);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
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
    onHoverRef.current(null);
    window.requestAnimationFrame(runtime.framePoints);
  }, [enabledRois, points]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    updatePointColors(runtime.pointCloud, visiblePointsRef.current, selectedCluster, runtime.background);
    onHoverRef.current(null);
  }, [selectedCluster]);

  useEffect(() => {
    if (resetKey > 0) runtimeRef.current?.framePoints();
  }, [resetKey]);

  return <div className="selectivity-atlas-viewer-shell" ref={mountRef} />;
}
