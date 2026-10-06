import { describe, expect, it } from 'vitest';
import { brainTooltipPosition } from './brainTooltipLayout';

describe('brain-map tooltip stage bounds', () => {
  const dimensions = { stageWidth: 400, stageHeight: 300, tooltipWidth: 235, tooltipHeight: 150 };

  it('keeps a small gap below and to the right of the hovered vertex when there is room', () => {
    expect(brainTooltipPosition({ ...dimensions, canvasX: 50, canvasY: 20 })).toEqual({ left: 62, top: 32 });
  });

  it('flips above and to the left near the right and bottom edges', () => {
    expect(brainTooltipPosition({ ...dimensions, canvasX: 388, canvasY: 290 })).toEqual({ left: 141, top: 128 });
  });

  it('clamps measured, wrapped tooltips on narrow canvases', () => {
    const stageWidth = 260;
    const stageHeight = 280;
    const tooltipWidth = 244;
    const tooltipHeight = 264;
    for (const [canvasX, canvasY] of [[0, 0], [130, 140], [259, 279], [-4, -4], [270, 290]]) {
      const position = brainTooltipPosition({ canvasX, canvasY, stageWidth, stageHeight, tooltipWidth, tooltipHeight });
      expect(position).toEqual({ left: 8, top: 8 });
      expect(position.left + tooltipWidth).toBeLessThanOrEqual(stageWidth - 8);
      expect(position.top + tooltipHeight).toBeLessThanOrEqual(stageHeight - 8);
    }
  });

  it('never produces negative positions when the stage is temporarily smaller than the tooltip', () => {
    expect(brainTooltipPosition({ canvasX: 0, canvasY: 0, stageWidth: 100, stageHeight: 90, tooltipWidth: 235, tooltipHeight: 150 }))
      .toEqual({ left: 0, top: 0 });
  });
});
