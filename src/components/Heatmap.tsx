import { useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent } from 'react';
import { extent, scaleBand, scaleSequential } from 'd3';
import type { AggregateHeatmapCell, Roi, SelectedHeatmapCell, SortDirection } from '../types/data';
import { inferModelCategory } from '../utils/modelTags';

type HeatmapProps = {
  cells: AggregateHeatmapCell[];
  models: string[];
  rois: Roi[];
  showScoreLabels: boolean;
  sortDirection: SortDirection;
  selectedCell: SelectedHeatmapCell | null;
  compareMode: boolean;
  compareCells: SelectedHeatmapCell[];
  comparableCellIds: Set<string>;
  onSortDirectionToggle: () => void;
  onSelectCell: (cell: SelectedHeatmapCell) => void;
  selectedRankingRoi?: Roi | null;
};

type TooltipState = {
  cell: AggregateHeatmapCell;
  x: number;
  y: number;
};

const margin = { top: 96, right: 18, bottom: 52, left: 142 };
const roiColumnWidth = 70;
const modelRowHeight = 22;
const sortToggleBoxSize = 30;

/**
 * Perceptually sequential scale: pale cream → warm amber → slate blue.
 * Works beautifully on the off-white surface background.
 */
function heatColor(t: number): string {
  const s = Math.max(0, Math.min(1, t));

  // Stops: [t, r, g, b]
  const stops: Array<[number, number, number, number]> = [
    [0.00,  242, 238, 228],  // near-white warm cream
    [0.20,  225, 210, 175],  // light sand
    [0.42,  210, 170, 100],  // warm amber
    [0.65,  152, 130, 190],  // muted lavender
    [0.82,   90,  96, 172],  // slate blue
    [1.00,   45,  55, 130],  // deep indigo
  ];

  let i = 0;
  while (i < stops.length - 2 && s > stops[i + 1][0]) i++;

  const [ta, ra, ga, ba] = stops[i];
  const [tb, rb, gb, bb] = stops[i + 1];
  const local = (s - ta) / (tb - ta);

  const r = Math.round(ra + (rb - ra) * local);
  const g = Math.round(ga + (gb - ga) * local);
  const b = Math.round(ba + (bb - ba) * local);
  return `rgb(${r},${g},${b})`;
}

function scoreTextColor(t: number): string {
  // light text for dark cells (upper half of scale)
  return t > 0.62 ? 'rgba(255,255,255,.88)' : 'rgba(28,25,23,.78)';
}

function formatScore(score: number | null): string {
  return score === null ? 'N/A' : score.toFixed(3);
}

function cellAriaLabel(cell: AggregateHeatmapCell, compareMode: boolean): string {
  const parts = [
    `ROI ${cell.roi}`,
    `model ${cell.model}`,
    `category ${inferModelCategory(cell.model)}`,
    `score ${formatScore(cell.score)}`,
    `rank ${cell.rankWithinRoi ?? 'unavailable'}`,
  ];
  return `${parts.join(', ')}. ${compareMode ? 'Press Enter to toggle compare selection.' : 'Press Enter to select.'}`;
}

export function Heatmap({
  cells,
  models,
  rois,
  showScoreLabels,
  sortDirection,
  selectedCell,
  compareMode,
  compareCells,
  comparableCellIds,
  onSortDirectionToggle,
  onSelectCell,
  selectedRankingRoi = null,
}: HeatmapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const hoveredCell = tooltip?.cell ?? null;

  const scoredCells = useMemo(() => cells.filter((c) => c.score !== null), [cells]);
  const [minScore = 0, maxScore = 1] = extent(scoredCells, (c) => c.score ?? undefined);
  const adjustedMax = minScore === maxScore ? maxScore + 1 : maxScore;

  const colorScale = scaleSequential<string>()
    .domain([minScore, adjustedMax])
    .interpolator((t) => heatColor(t));

  const width = Math.max(430, margin.left + margin.right + rois.length * roiColumnWidth);
  const heatmapHeight = Math.max(220, models.length * modelRowHeight);
  const height = margin.top + margin.bottom + heatmapHeight;
  const innerWidth = width - margin.left - margin.right;
  const sortToggleX = margin.left - sortToggleBoxSize - 18;
  const sortToggleY = margin.top - sortToggleBoxSize + 2;

  const x = scaleBand<string>().domain(rois).range([0, innerWidth]).padding(0.12);
  const y = scaleBand<string>().domain(models).range([0, heatmapHeight]).padding(0.08);
  const canShowLabels = showScoreLabels && x.bandwidth() >= 44 && y.bandwidth() >= 18;
  const labelSize = Math.max(8, Math.min(11, Math.min(x.bandwidth() * 0.2, y.bandwidth() * 0.55)));
  const legendVals = [minScore, (minScore + adjustedMax) / 2, adjustedMax];
  const legendSteps = Array.from({ length: 48 }, (_, i) => minScore + ((adjustedMax - minScore) * i) / 47);

  function updateTooltip(cell: AggregateHeatmapCell, event: MouseEvent<SVGRectElement>) {
    const bounds = containerRef.current?.getBoundingClientRect();
    setTooltip({
      cell,
      x: bounds ? event.clientX - bounds.left + 14 : event.clientX,
      y: bounds ? event.clientY - bounds.top + 14 : event.clientY,
    });
  }

  function updateFocusTooltip(cell: AggregateHeatmapCell, xp: number, yp: number) {
    setTooltip({
      cell,
      x: margin.left + xp + Math.max(x.bandwidth() / 2, 12),
      y: margin.top + yp + Math.max(y.bandwidth() / 2, 12),
    });
  }

  function handleCellKeyDown(cell: AggregateHeatmapCell, event: KeyboardEvent<SVGRectElement>, isComparable: boolean) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (compareMode && !isComparable) return;
      onSelectCell(cell);
    }
  }

  function handleSortKeyDown(event: KeyboardEvent<SVGGElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSortDirectionToggle();
    }
  }

  return (
    <section className="heatmap-area" aria-label="Heatmap area" ref={containerRef}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Overview</p>
          <h2>Aggregate score by model &amp; ROI</h2>
        </div>
        <span>{cells.length} cells</span>
      </div>

      <div className="heatmap-scroll" onMouseLeave={() => setTooltip(null)}>
        {cells.length === 0 ? (
          <div className="heatmap-empty-state">
            <h3>No models match</h3>
            <p>Clear the search or reset controls to restore the heatmap.</p>
          </div>
        ) : (
          <svg className="heatmap-svg" viewBox={`0 0 ${width} ${height}`} role="img">
            <title>Model × ROI aggregate score heatmap</title>

            {/* ── Top legend ── */}
            <g className="heatmap-legend" transform={`translate(${margin.left},${margin.top - 70})`}>
              {legendSteps.map((v, i) => (
                <rect key={v} x={i * 4} y={0} width={4} height={10} fill={colorScale(v)} />
              ))}
              {legendVals.map((v, i) => (
                <text key={v} x={i * 94} y={30} textAnchor={i === 2 ? 'end' : 'start'}>
                  {v.toFixed(2)}
                </text>
              ))}
              <text x={214} y={10} dominantBaseline="middle">Aggregate score</text>
              {compareMode && (
                <g className="disabled-cell-legend" transform="translate(328,0)">
                  <rect width={16} height={10} rx={3} />
                  <text x={24} y={10} dominantBaseline="middle">Not comparable</text>
                </g>
              )}
            </g>

            {/* ── Sort toggle ── */}
            <g
              className="heatmap-corner-sort"
              role="button"
              tabIndex={0}
              aria-label={`Sort ${sortDirection === 'desc' ? 'descending' : 'ascending'} — click to toggle`}
              transform={`translate(${sortToggleX},${sortToggleY})`}
              onClick={onSortDirectionToggle}
              onKeyDown={handleSortKeyDown}
            >
              <rect width={sortToggleBoxSize} height={sortToggleBoxSize} rx={6} />
              <line
                x1={sortToggleBoxSize / 2} y1={sortDirection === 'desc' ? 12 : 22}
                x2={sortToggleBoxSize / 2} y2={sortDirection === 'desc' ? 22 : 12}
              />
              <polyline
                points={
                  sortDirection === 'desc'
                    ? `${sortToggleBoxSize / 2 - 4},18 ${sortToggleBoxSize / 2},22 ${sortToggleBoxSize / 2 + 4},18`
                    : `${sortToggleBoxSize / 2 - 4},16 ${sortToggleBoxSize / 2},12 ${sortToggleBoxSize / 2 + 4},16`
                }
              />
            </g>

            <g transform={`translate(${margin.left},${margin.top})`}>

              {/* ── ROI column highlight ── */}
              {selectedRankingRoi && (
                <g style={{ pointerEvents: 'none' }}>
                  <rect
                    className="roi-column-highlight"
                    x={(x(selectedRankingRoi) ?? 0) - 3}
                    y={-3}
                    width={(x.bandwidth() || 0) + 6}
                    height={heatmapHeight + 6}
                    fill="#FEF3C7"
                    fillOpacity={0.55}
                  />
                  <circle cx={(x(selectedRankingRoi) ?? 0) - 3}                     cy={-3}              r="5" className="roi-corner-dot" />
                  <circle cx={(x(selectedRankingRoi) ?? 0) - 3}                     cy={heatmapHeight + 3} r="5" className="roi-corner-dot" />
                  <circle cx={(x(selectedRankingRoi) ?? 0) + (x.bandwidth() || 0) + 3} cy={-3}              r="5" className="roi-corner-dot" />
                  <circle cx={(x(selectedRankingRoi) ?? 0) + (x.bandwidth() || 0) + 3} cy={heatmapHeight + 3} r="5" className="roi-corner-dot" />
                </g>
              )}

              {/* ── Crosshair highlight on hover ── */}
              {hoveredCell && (
                <>
                  <rect className="heatmap-highlight" x={0}                    y={y(hoveredCell.model) ?? 0} width={innerWidth}    height={y.bandwidth()} />
                  <rect className="heatmap-highlight" x={x(hoveredCell.roi) ?? 0} y={0}                   width={x.bandwidth()} height={heatmapHeight} />
                </>
              )}

              {/* ── Cells ── */}
              {cells.map((cell) => {
                const xp = x(cell.roi) ?? 0;
                const yp = y(cell.model) ?? 0;
                const cmpIdx = compareCells.findIndex((c) => c.id === cell.id);
                const isCompareSelected = cmpIdx !== -1;
                const isSelected = !compareMode && selectedCell?.id === cell.id;
                const isComparable = comparableCellIds.has(cell.id);
                const isDisabled = compareMode && !isComparable;
                const normalised = cell.score === null ? 0 : (cell.score - minScore) / (adjustedMax - minScore);
                const fill = cell.score === null ? '#EDE9E3' : colorScale(cell.score);

                return (
                  <g key={cell.id}>
                    <rect
                      className={[
                        'heatmap-cell',
                        isSelected ? 'selected' : '',
                        isCompareSelected ? 'compare-selected' : '',
                        isDisabled ? 'compare-disabled' : '',
                      ].filter(Boolean).join(' ')}
                      role="button"
                      tabIndex={0}
                      aria-disabled={isDisabled}
                      aria-label={
                        isDisabled
                          ? `${cellAriaLabel(cell, compareMode)} No image-level data for comparison.`
                          : cellAriaLabel(cell, compareMode)
                      }
                      x={xp} y={yp}
                      width={x.bandwidth()} height={y.bandwidth()}
                      rx={5}
                      fill={fill}
                      onMouseEnter={(e) => updateTooltip(cell, e)}
                      onMouseMove={(e) => updateTooltip(cell, e)}
                      onFocus={() => updateFocusTooltip(cell, xp, yp)}
                      onBlur={() => setTooltip(null)}
                      onKeyDown={(e) => handleCellKeyDown(cell, e, isComparable)}
                      onClick={() => { if (!isDisabled) onSelectCell(cell); }}
                    />
                    {isCompareSelected && (
                      <text className="compare-cell-marker" x={xp + 6} y={yp + 14}>
                        {cmpIdx === 0 ? 'A' : 'B'}
                      </text>
                    )}
                    {canShowLabels && cell.score !== null && (
                      <text
                        className="cell-score-label"
                        x={xp + x.bandwidth() / 2}
                        y={yp + y.bandwidth() / 2}
                        fill={scoreTextColor(normalised)}
                        style={{ fontSize: `${labelSize}px` }}
                      >
                        {cell.score.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* ── Y-axis (model names) ── */}
              <g className="axis-labels y-axis-labels">
                {models.map((model) => (
                  <text key={model} x={-14} y={(y(model) ?? 0) + y.bandwidth() / 2} textAnchor="end" dominantBaseline="middle">
                    {model}
                  </text>
                ))}
              </g>

              {/* ── X-axis (ROI names) ── */}
              <g className="axis-labels x-axis-labels">
                {rois.map((roi) => (
                  <text key={roi} x={(x(roi) ?? 0) + x.bandwidth() / 2} y={-14} textAnchor="middle">
                    {roi.toUpperCase()}
                  </text>
                ))}
              </g>
            </g>

            {/* ── Bottom legend ── */}
            <g className="heatmap-legend" transform={`translate(${margin.left},${height - 42})`}>
              {legendSteps.map((v, i) => (
                <rect key={v} x={i * 4} y={0} width={4} height={10} fill={colorScale(v)} />
              ))}
              {legendVals.map((v, i) => (
                <text key={v} x={i * 94} y={30} textAnchor={i === 2 ? 'end' : 'start'}>
                  {v.toFixed(2)}
                </text>
              ))}
              <text x={214} y={10} dominantBaseline="middle">Aggregate score</text>
              {compareMode && (
                <g className="disabled-cell-legend" transform="translate(328,0)">
                  <rect width={16} height={10} rx={3} />
                  <text x={24} y={10} dominantBaseline="middle">Not comparable</text>
                </g>
              )}
            </g>
          </svg>
        )}
      </div>

      {/* ── Tooltip ── */}
      {tooltip && (
        <div className="heatmap-tooltip" role="tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
          <dl>
            <div><dt>ROI</dt><dd>{tooltip.cell.roi.toUpperCase()}</dd></div>
            <div>
              <dt>Model</dt>
              <dd>{tooltip.cell.model} <span className="model-tag">{inferModelCategory(tooltip.cell.model)}</span></dd>
            </div>
            <div>
              <dt>Score</dt>
              <dd>
                {compareMode && !comparableCellIds.has(tooltip.cell.id)
                  ? 'No CSV data'
                  : formatScore(tooltip.cell.score)}
              </dd>
            </div>
            <div><dt>ROI rank</dt><dd>{tooltip.cell.rankWithinRoi === null ? 'N/A' : `#${tooltip.cell.rankWithinRoi}`}</dd></div>
            <div><dt>Overall</dt><dd>{formatScore(tooltip.cell.overallScore)}</dd></div>
          </dl>
        </div>
      )}
    </section>
  );
}
