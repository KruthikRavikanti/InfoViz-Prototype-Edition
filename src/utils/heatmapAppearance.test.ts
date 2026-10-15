import { describe, expect, it } from 'vitest';
import {
  clampHeatmapTooltip,
  getHeatmapLayout,
  heatmapScoreColor,
  heatmapScoreTextColor,
  SCOREBOARD_HEATMAP_COLORS,
} from './heatmapAppearance';

describe('scoreboard-aligned heatmap appearance', () => {
  it('matches all three quantitative scoreboard palette stops exactly', () => {
    expect(SCOREBOARD_HEATMAP_COLORS).toEqual(['#ede9d8', '#c4b4cc', '#8966a3']);
    expect(heatmapScoreColor(0)).toBe('rgb(237, 233, 216)');
    expect(heatmapScoreColor(0.5)).toBe('rgb(196, 180, 204)');
    expect(heatmapScoreColor(1)).toBe('rgb(137, 102, 163)');
  });

  it('uses fixed normalization so searching or sorting cannot recolor a score', () => {
    expect(heatmapScoreColor(0.5)).toBe('rgb(196, 180, 204)');
    // No filtered score domain or ratio recalculation enters this function.
    expect(heatmapScoreColor(0.9)).toBe('rgb(149, 118, 171)');
  });

  it('clamps out-of-domain values and chooses readable text for the new ramp', () => {
    expect(heatmapScoreColor(-3)).toBe(heatmapScoreColor(0));
    expect(heatmapScoreColor(3)).toBe(heatmapScoreColor(1));
    expect(heatmapScoreTextColor(0)).toBe('#000000');
    expect(heatmapScoreTextColor(0.5)).toBe('#000000');
    expect(heatmapScoreTextColor(1)).toBe('#ffffff');
  });

  it('uses a narrow intrinsic canvas and fixed square cell geometry regardless of filtering', () => {
    expect(getHeatmapLayout(26, 3)).toEqual({
      margin: { top: 38, right: 12, bottom: 8, left: 166 },
      innerWidth: 76,
      innerHeight: 674,
      width: 254,
      height: 720,
    });
    expect(getHeatmapLayout(1, 3)).toMatchObject({ innerWidth: 76, innerHeight: 24, width: 254, height: 70 });
  });

  it('keeps tooltip bounds inside the viewport instead of inside the clipped panel', () => {
    expect(clampHeatmapTooltip(390, 810, 250, 180, 390, 844)).toEqual({ x: 132, y: 656 });
    expect(clampHeatmapTooltip(-20, -30, 250, 180, 390, 844)).toEqual({ x: 8, y: 8 });
  });
});
