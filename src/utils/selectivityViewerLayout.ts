export type SelectivityPosition = readonly [number, number, number];

export const SELECTIVITY_DEFAULT_CAMERA_DIRECTION: SelectivityPosition = [1, 0.55, 1.35];

function dot(a: SelectivityPosition, b: SelectivityPosition): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalized(vector: SelectivityPosition): SelectivityPosition {
  const length = Math.hypot(...vector);
  if (!Number.isFinite(length) || length <= 0) throw new Error('Invalid selectivity camera direction.');
  return vector.map((coordinate) => coordinate / length) as [number, number, number];
}

function cross(a: SelectivityPosition, b: SelectivityPosition): SelectivityPosition {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** The full model, not the visible ROI subset, defines the reset target. */
export function selectivityBoundsCenter(positions: readonly SelectivityPosition[]): SelectivityPosition {
  if (positions.length === 0) return [0, 0, 0];
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const position of positions) {
    for (let axis = 0; axis < 3; axis += 1) {
      if (!Number.isFinite(position[axis])) throw new Error('Invalid selectivity point coordinate.');
      min[axis] = Math.min(min[axis], position[axis]);
      max[axis] = Math.max(max[axis], position[axis]);
    }
  }
  return min.map((coordinate, axis) => (coordinate + max[axis]) / 2) as [number, number, number];
}

/** Fit raw UMAP coordinates in the actual oblique camera basis, including depth. */
export function selectivityFrameDistance(
  positions: readonly SelectivityPosition[],
  center: SelectivityPosition,
  aspect: number,
  verticalFovDegrees = 38,
  direction: SelectivityPosition = SELECTIVITY_DEFAULT_CAMERA_DIRECTION,
): number {
  if (!Number.isFinite(aspect) || aspect <= 0
    || !Number.isFinite(verticalFovDegrees) || verticalFovDegrees <= 0 || verticalFovDegrees >= 180) {
    throw new Error('Invalid selectivity camera bounds.');
  }
  const offset = normalized(direction);
  const look: SelectivityPosition = [-offset[0], -offset[1], -offset[2]];
  // OrbitControls avoids the poles; retain a valid basis for mathematical callers too.
  const referenceUp: SelectivityPosition = Math.abs(look[1]) > 0.999999 ? [0, 0, 1] : [0, 1, 0];
  const right = normalized(cross(look, referenceUp));
  const up = normalized(cross(right, look));
  const tanHalfFov = Math.tan(verticalFovDegrees * Math.PI / 360);
  const frameFraction = 0.78;
  let distance = 2;
  for (const position of positions) {
    const relative: SelectivityPosition = [
      position[0] - center[0], position[1] - center[1], position[2] - center[2],
    ];
    const depthOffset = dot(relative, look);
    distance = Math.max(
      distance,
      Math.abs(dot(relative, right)) / (tanHalfFov * aspect * frameFraction) - depthOffset,
      Math.abs(dot(relative, up)) / (tanHalfFov * frameFraction) - depthOffset,
      0.1 - depthOffset,
    );
  }
  return distance;
}

/** Preserve the zoom relative to a full-model fit when the viewport aspect changes. */
export function resizedSelectivityCameraDistance(distance: number, oldFit: number, newFit: number): number {
  if (![distance, oldFit, newFit].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error('Invalid selectivity resize distance.');
  }
  return distance * newFit / oldFit;
}

/** Orbit the camera about the data target without rotating translated UMAP coordinates. */
export function orbitSelectivityCamera(
  position: SelectivityPosition,
  target: SelectivityPosition,
  azimuthDelta: number,
  polarDelta: number,
): SelectivityPosition {
  const offset: SelectivityPosition = [
    position[0] - target[0], position[1] - target[1], position[2] - target[2],
  ];
  const radius = Math.hypot(...offset);
  if (radius <= 0 || !Number.isFinite(radius)) throw new Error('Invalid selectivity orbit radius.');
  const theta = Math.atan2(offset[0], offset[2]) + azimuthDelta;
  const phi = Math.min(Math.PI - 1e-5, Math.max(1e-5, Math.acos(Math.max(-1, Math.min(1, offset[1] / radius))) + polarDelta));
  return [
    target[0] + radius * Math.sin(phi) * Math.sin(theta),
    target[1] + radius * Math.cos(phi),
    target[2] + radius * Math.sin(phi) * Math.cos(theta),
  ];
}

/** Keyboard navigation uses the same visible-point array as rendering and picking. */
export function selectivityNavigationIndex(count: number, current: number | null, key: string): number | null {
  if (!Number.isInteger(count) || count <= 0) return null;
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  const previous = key === '[' || key === 'PageUp';
  const next = key === ']' || key === 'PageDown';
  if (!previous && !next) return null;
  if (current === null || current < 0 || current >= count) return previous ? count - 1 : 0;
  return (current + (previous ? -1 : 1) + count) % count;
}

/** Camera quaternion converts world-positive UMAP axes into camera-local directions. */
export function selectivityOrientationAxes(
  cameraQuaternion: readonly [number, number, number, number],
): readonly SelectivityPosition[] {
  const length = Math.hypot(...cameraQuaternion);
  if (length <= 0 || !Number.isFinite(length)) throw new Error('Invalid selectivity camera quaternion.');
  const [x, y, z, w] = cameraQuaternion.map((coordinate) => coordinate / length);
  // Rows of the camera world rotation matrix are the columns of its inverse.
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}
