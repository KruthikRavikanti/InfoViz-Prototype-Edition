export type AxisBounds = { min: number; max: number };

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
