import { scaleLinear } from 'd3';
// Matches Cortex's quantitative scoreboard heatmapOverview/heatmapDetail ramp.
// Input is the source file's stored ceiling-normalized value. Raw correlation
// remains unchanged in labels and rankings; only colors are clipped to [0, 1].
export const SCOREBOARD_HEATMAP_COLORS = ['#ede9d8', '#c4b4cc', '#8966a3'] as const;
export const SCOREBOARD_HEATMAP_GRADIENT = `linear-gradient(90deg, ${SCOREBOARD_HEATMAP_COLORS[0]} 0%, ${SCOREBOARD_HEATMAP_COLORS[1]} 50%, ${SCOREBOARD_HEATMAP_COLORS[2]} 100%)`;

const scoreboardRamp = scaleLinear<string>()
  .domain([0, 0.5, 1])
  .range([...SCOREBOARD_HEATMAP_COLORS])
  .clamp(true);

export function heatmapScoreColor(normalizedScore: number): string {
  return scoreboardRamp(normalizedScore);
}

export function heatmapScoreTextColor(normalizedScore: number): string {
  const channels = heatmapScoreColor(normalizedScore).match(/\d+/g)?.map(Number) ?? [237, 233, 216];
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
