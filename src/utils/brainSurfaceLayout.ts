export type AxisBounds = { min: number; max: number };

// Match Cortex's UploadBrainViewer display presets. These affect only the
// camera, never the registered mesh or its voxel-to-vertex correspondence.
export const BRAIN_SURFACE_VIEWS = {
  overview: { label: 'Overview', direction: [0.9, 0.28, 1.35], up: [0, 1, 0] },
  lateral: { label: 'Right side', direction: [1, 0, 0], up: [0, 1, 0] },
  medial: { label: 'Left side', direction: [-1, 0, 0], up: [0, 1, 0] },
  ventral: { label: 'Ventral', direction: [0, -1, 0], up: [0, 0, -1] },
  dorsal: { label: 'Dorsal', direction: [0, 1, 0], up: [0, 0, -1] },
} as const;
export type BrainSurfaceView = keyof typeof BRAIN_SURFACE_VIEWS;

/** Fit the entire cortical bounding sphere along the narrower camera dimension. */
export function brainSurfaceCameraDistance(radius: number, aspect: number, verticalFovDegrees = 34): number {
  if (
    !Number.isFinite(radius) || radius <= 0
    || !Number.isFinite(aspect) || aspect <= 0
    || !Number.isFinite(verticalFovDegrees) || verticalFovDegrees <= 0 || verticalFovDegrees >= 180
  ) throw new Error('Brain surface camera bounds are invalid.');
  const verticalFov = verticalFovDegrees * Math.PI / 180;
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * aspect);
  return radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.1;
}

export function inflatedHemisphereTranslations(
  left: AxisBounds,
  right: AxisBounds,
  gapMillimeters = 8,
): { lh: number; rh: number } {
  if (
    !Number.isFinite(left.min)
    || !Number.isFinite(left.max)
    || !Number.isFinite(right.min)
    || !Number.isFinite(right.max)
    || left.max <= left.min
    || right.max <= right.min
    || !Number.isFinite(gapMillimeters)
    || gapMillimeters <= 0
  ) {
    throw new Error('Inflated hemisphere bounds are invalid.');
  }

  const leftWidth = left.max - left.min;
  const rightWidth = right.max - right.min;
  const leftCenter = (left.min + left.max) / 2;
  const rightCenter = (right.min + right.max) / 2;
  const targetLeftCenter = -(gapMillimeters / 2 + leftWidth / 2);
  const targetRightCenter = gapMillimeters / 2 + rightWidth / 2;
  return {
    lh: targetLeftCenter - leftCenter,
    rh: targetRightCenter - rightCenter,
  };
}
