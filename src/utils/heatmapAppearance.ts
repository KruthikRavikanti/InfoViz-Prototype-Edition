import { scaleLinear } from 'd3';
import type { AggregateHeatmapCell } from '../types/data';

export type HeatmapScoreDomain = readonly [number, number];

// Matches Cortex's quantitative scoreboard heatmapOverview/heatmapDetail ramp.
// These are palette stops, not ceiling-normalized scores: the legend always
// reports the aggregate alignment scores actually supplied by InfoViz.
export const SCOREBOARD_HEATMAP_COLORS = ['#ede9d8', '#c4b4cc', '#8966a3'] as const;
export const SCOREBOARD_HEATMAP_GRADIENT = `linear-gradient(90deg, ${SCOREBOARD_HEATMAP_COLORS[0]} 0%, ${SCOREBOARD_HEATMAP_COLORS[1]} 50%, ${SCOREBOARD_HEATMAP_COLORS[2]} 100%)`;

const scoreboardRamp = scaleLinear<string>()
  .domain([0, 0.5, 1])
  .range([...SCOREBOARD_HEATMAP_COLORS])
  .clamp(true);

export function getHeatmapScoreDomain(cells: readonly AggregateHeatmapCell[]): HeatmapScoreDomain {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const cell of cells) {
    if (cell.score === null || !Number.isFinite(cell.score)) continue;
    minimum = Math.min(minimum, cell.score);
    maximum = Math.max(maximum, cell.score);
  }
  return minimum === Infinity ? [0, 1] : [minimum, maximum];
}

export function normalizeHeatmapScore(score: number, [minimum, maximum]: HeatmapScoreDomain): number {
  // A constant data set has no low/high distinction. Do not invent a larger
  // score for its legend merely to give the color scale a nonzero width.
  if (minimum === maximum) return 0.5;
  return Math.max(0, Math.min(1, (score - minimum) / (maximum - minimum)));
}

export function heatmapScoreColor(score: number, domain: HeatmapScoreDomain): string {
  return scoreboardRamp(normalizeHeatmapScore(score, domain));
}

export function heatmapScoreTextColor(score: number, domain: HeatmapScoreDomain): string {
  const channels = heatmapScoreColor(score, domain).match(/\d+/g)?.map(Number) ?? [237, 233, 216];
  const linear = channels.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  const luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  // Choose the stronger of black/white instead of assuming the old blue ramp's
  // threshold remains readable on the scoreboard's lighter purple colors.
  return (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) ? '#000000' : '#ffffff';
}

export const HEATMAP_LAYOUT = {
  margin: { top: 38, right: 12, bottom: 8, left: 166 },
  columnPitch: 26,
  rowPitch: 26,
  cellGap: 2,
} as const;

export function getHeatmapLayout(modelCount: number, roiCount: number) {
  const { margin, columnPitch, rowPitch, cellGap } = HEATMAP_LAYOUT;
  const innerWidth = Math.max(0, roiCount * columnPitch - cellGap);
  const innerHeight = Math.max(0, modelCount * rowPitch - cellGap);
  return {
    margin,
    innerWidth,
    innerHeight,
    width: margin.left + innerWidth + margin.right,
    height: margin.top + innerHeight + margin.bottom,
  };
}

export function clampHeatmapTooltip(
  x: number,
  y: number,
  tooltipWidth: number,
  tooltipHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): { x: number; y: number } {
  return {
    x: Math.max(8, Math.min(x, viewportWidth - tooltipWidth - 8)),
    y: Math.max(8, Math.min(y, viewportHeight - tooltipHeight - 8)),
  };
}
