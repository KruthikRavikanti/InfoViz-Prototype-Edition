import { describe, expect, it } from 'vitest';
import type { AggregateHeatmapCell } from '../types/data';
import {
  clampHeatmapTooltip,
  getHeatmapLayout,
  getHeatmapScoreDomain,
  heatmapScoreColor,
  heatmapScoreTextColor,
  normalizeHeatmapScore,
  SCOREBOARD_HEATMAP_COLORS,
} from './heatmapAppearance';

function scoredCell(score: number | null): AggregateHeatmapCell {
  return { id: `ffa__${score}`, roi: 'ffa', model: `${score}`, score, overallScore: score, rankWithinRoi: null };
}

describe('scoreboard-aligned heatmap appearance', () => {
  it('matches all three quantitative scoreboard palette stops exactly', () => {
    expect(SCOREBOARD_HEATMAP_COLORS).toEqual(['#ede9d8', '#c4b4cc', '#8966a3']);
    expect(heatmapScoreColor(-0.28, [-0.28, 0.86])).toBe('rgb(237, 233, 216)');
    expect(heatmapScoreColor(0.29, [-0.28, 0.86])).toBe('rgb(196, 180, 204)');
    expect(heatmapScoreColor(0.86, [-0.28, 0.86])).toBe('rgb(137, 102, 163)');
  });

  it('derives the actual aggregate score domain without ceiling normalization', () => {
    expect(getHeatmapScoreDomain([scoredCell(-0.281), scoredCell(0.86), scoredCell(null), scoredCell(NaN)]))
      .toEqual([-0.281, 0.86]);
    expect(getHeatmapScoreDomain([])).toEqual([0, 1]);
  });

  it('can keep the same score color across filtering by using the full domain', () => {
    const fullDomain = getHeatmapScoreDomain([scoredCell(-0.28), scoredCell(0.29), scoredCell(0.86)]);
    const beforeFiltering = heatmapScoreColor(0.29, fullDomain);
    expect(heatmapScoreColor(0.29, fullDomain)).toBe(beforeFiltering);
    expect(normalizeHeatmapScore(0.29, fullDomain)).toBeCloseTo(0.5);
  });

  it('handles constant scores without fabricating a higher legend endpoint', () => {
    const domain = getHeatmapScoreDomain([scoredCell(0.812)]);
    expect(domain).toEqual([0.812, 0.812]);
    expect(heatmapScoreColor(0.812, domain)).toBe('rgb(196, 180, 204)');
  });

  it('clamps out-of-domain values and chooses readable text for the new ramp', () => {
    expect(heatmapScoreColor(-3, [0, 1])).toBe(heatmapScoreColor(0, [0, 1]));
    expect(heatmapScoreColor(3, [0, 1])).toBe(heatmapScoreColor(1, [0, 1]));
    expect(heatmapScoreTextColor(0, [0, 1])).toBe('#000000');
    expect(heatmapScoreTextColor(0.5, [0, 1])).toBe('#000000');
    expect(heatmapScoreTextColor(1, [0, 1])).toBe('#ffffff');
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
