/** Flip away from canvas edges, then keep the measured tooltip inside the stage. */
export function brainTooltipPosition({
  canvasX,
  canvasY,
  stageWidth,
  stageHeight,
  tooltipWidth,
  tooltipHeight,
}: {
  canvasX: number;
  canvasY: number;
  stageWidth: number;
  stageHeight: number;
  tooltipWidth: number;
  tooltipHeight: number;
}): { left: number; top: number } {
  const gap = 12;
  const padding = 8;
  function coordinate(pointer: number, extent: number, size: number) {
    const inset = Math.min(padding, Math.max(0, (extent - size) / 2));
    const maximum = Math.max(inset, extent - size - inset);
    const preferred = pointer + gap;
    const flipped = preferred > maximum ? pointer - size - gap : preferred;
    return Math.max(inset, Math.min(maximum, flipped));
  }
  return {
    left: coordinate(canvasX, stageWidth, tooltipWidth),
    top: coordinate(canvasY, stageHeight, tooltipHeight),
  };
}
