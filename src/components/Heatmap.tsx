import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent } from 'react';
import { createPortal } from 'react-dom';
import { scaleBand } from 'd3';
import type { AggregateHeatmapCell, Roi, SelectedHeatmapCell, SortDirection } from '../types/data';
import { inferModelCategory } from '../utils/modelTags';
import {
  clampHeatmapTooltip,
  getHeatmapLayout,
  HEATMAP_LAYOUT,
  heatmapScoreColor,
  heatmapScoreTextColor,
  SCOREBOARD_HEATMAP_GRADIENT,
} from '../utils/heatmapAppearance';

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
  sessionCellIds: Set<string>;
  onSortDirectionToggle: () => void;
  onSelectCell: (cell: SelectedHeatmapCell) => void;
  selectedRankingRoi?: Roi | null;
};

type TooltipState = {
  cell: AggregateHeatmapCell;
  x: number;
  y: number;
};

const sortToggleBoxSize = 30;

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
  sessionCellIds,
  onSortDirectionToggle,
  onSelectCell,
  selectedRankingRoi = null,
}: HeatmapProps) {
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const hoveredCell = tooltip?.cell ?? null;
  const { margin, width, height, innerWidth, innerHeight: heatmapHeight } = getHeatmapLayout(models.length, rois.length);
  const sortToggleX = margin.left - sortToggleBoxSize - 18;
  const sortToggleY = margin.top - sortToggleBoxSize - 4;

  const x = scaleBand<string>().domain(rois)
    .range([0, rois.length === 1 ? HEATMAP_LAYOUT.columnPitch : innerWidth])
    .paddingInner(HEATMAP_LAYOUT.cellGap / HEATMAP_LAYOUT.columnPitch).align(0);
  const y = scaleBand<string>().domain(models)
    .range([0, models.length === 1 ? HEATMAP_LAYOUT.rowPitch : heatmapHeight])
    .paddingInner(HEATMAP_LAYOUT.cellGap / HEATMAP_LAYOUT.rowPitch).align(0);
  const canShowLabels = showScoreLabels && x.bandwidth() >= 23.5 && y.bandwidth() >= 20;
  const labelSize = 9.5;
  const legendVals = [0, 0.5, 1];
  const hasTooltip = tooltip !== null;

  useLayoutEffect(() => {
    if (!tooltip || !tooltipRef.current) return;
    const bounds = tooltipRef.current.getBoundingClientRect();
    const position = clampHeatmapTooltip(tooltip.x, tooltip.y, bounds.width, bounds.height, window.innerWidth, window.innerHeight);
    if (position.x !== tooltip.x || position.y !== tooltip.y) {
      setTooltip((current) => current ? { ...current, ...position } : null);
    }
  }, [tooltip]);

  useEffect(() => {
    if (!hasTooltip) return;
    const dismiss = () => setTooltip(null);
    window.addEventListener('scroll', dismiss, { capture: true, passive: true });
    window.addEventListener('resize', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
    };
  }, [hasTooltip]);

  function updateTooltip(cell: AggregateHeatmapCell, event: MouseEvent<SVGRectElement>) {
    setTooltip({
      cell,
      x: event.clientX + 14,
      y: event.clientY + 14,
    });
  }

  function updateFocusTooltip(cell: AggregateHeatmapCell, target: SVGRectElement) {
    const bounds = target.getBoundingClientRect();
    setTooltip({
      cell,
      x: bounds.right + 8,
      y: bounds.top,
    });
  }

  function handleCellKeyDown(cell: AggregateHeatmapCell, event: KeyboardEvent<SVGRectElement>, isComparable: boolean) {
    if (event.key === 'Escape') setTooltip(null);
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
    <section className="heatmap-area" aria-label="Heatmap area">
      <div className="section-heading">
        <div>
          <h2>Model alignment</h2>
        </div>
        <span>{models.filter((model) => model !== 'ceiling').length} models{models.includes('ceiling') ? ' + ceiling' : ''}</span>
      </div>

      <p className="heatmap-benchmark-context">NSD-1000 training · Murty185 evaluation · Pearson r</p>
      <div className="heatmap-color-key" aria-label="Score / noise ceiling color scale from 0 to 1" title="Numbers and ranks use raw Pearson r. Colors use the scoreboard's stored score / noise ceiling values, clipped to 0–1. The noise ceiling is not ranked as a model.">
        <p className="heatmap-color-key-title">Color: score / noise ceiling</p>
        <div className="heatmap-color-key-bar" style={{ background: SCOREBOARD_HEATMAP_GRADIENT }} />
        <div className="heatmap-color-key-values">
          {legendVals.map((value) => <span key={value}>{value.toFixed(2)}</span>)}
        </div>
        {compareMode && (
          <div className="heatmap-color-key-unavailable"><span aria-hidden="true" />Not comparable</div>
        )}
      </div>

      <div className="heatmap-scroll" onMouseLeave={() => setTooltip(null)}>
        {cells.length === 0 ? (
          <div className="heatmap-empty-state">
            <h3>No models match</h3>
            <p>Clear the search or reset controls.</p>
          </div>
        ) : (
          <svg className="heatmap-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img">
            <title>Model × ROI aggregate score heatmap</title>

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
                    fill="var(--accent-light)"
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
                const hasSessionEvidence = sessionCellIds.has(cell.id);
                const fill = cell.score === null || cell.normalizedScore == null
                  ? 'var(--bg-subtle)' : heatmapScoreColor(cell.normalizedScore);

                return (
                  <g key={cell.id}>
                    <rect
                      className={[
                        'heatmap-cell',
                        cell.model === 'ceiling' ? 'ceiling-reference' : '',
                        isSelected ? 'selected' : '',
                        isCompareSelected ? 'compare-selected' : '',
                        isDisabled ? 'compare-disabled' : '',
                        hasSessionEvidence ? 'session-evidence' : '',
                      ].filter(Boolean).join(' ')}
                      data-cell-id={cell.id}
                      data-score={cell.score ?? ''}
                      data-normalized-score={cell.normalizedScore ?? ''}
                      data-session-evidence={hasSessionEvidence ? 'true' : undefined}
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
                      rx={3}
                      fill={fill}
                      onMouseEnter={(e) => updateTooltip(cell, e)}
                      onMouseMove={(e) => updateTooltip(cell, e)}
                      onFocus={(event) => updateFocusTooltip(cell, event.currentTarget)}
                      onBlur={() => setTooltip(null)}
                      onKeyDown={(e) => handleCellKeyDown(cell, e, isComparable)}
                      onClick={() => { if (!isDisabled) onSelectCell(cell); }}
                    />
                    {isCompareSelected && (
                      <text className="compare-cell-marker" x={xp + 4} y={yp + 9} style={{ fontSize: '8px' }}>
                        {cmpIdx === 0 ? 'A' : 'B'}
                      </text>
                    )}
                    {hasSessionEvidence && (
                      <circle
                        className="session-cell-marker"
                        cx={xp + x.bandwidth() - 5}
                        cy={yp + 5}
                        r="2.8"
                        aria-hidden="true"
                      />
                    )}
                    {canShowLabels && cell.score !== null && (
                      <text
                        className="cell-score-label"
                        x={xp + x.bandwidth() / 2}
                        y={yp + y.bandwidth() / 2}
                        fill={heatmapScoreTextColor(cell.normalizedScore ?? 0)}
                        style={{ fontSize: `${labelSize}px` }}
                      >
                        {cell.score.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* ── Y-axis (model names) ── */}
              {models.includes('ceiling') && models.length > 1 && (
                <line className="heatmap-ceiling-divider" x1={-margin.left} x2={innerWidth}
                  y1={(y('ceiling') ?? 0) + y.bandwidth() + 1}
                  y2={(y('ceiling') ?? 0) + y.bandwidth() + 1} />
              )}
              <g className="axis-labels y-axis-labels">
                {models.map((model) => (
                  <text key={model} x={-14} y={(y(model) ?? 0) + y.bandwidth() / 2} textAnchor="end" dominantBaseline="middle">
                    {model === 'ceiling' ? 'Noise ceiling' : model}
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

          </svg>
        )}
      </div>

      {/* ── Tooltip ── */}
      {tooltip && typeof document !== 'undefined' && createPortal(
        <div className="heatmap-tooltip" role="tooltip" ref={tooltipRef} style={{ left: tooltip.x, top: tooltip.y }}>
          <dl>
            <div><dt>ROI</dt><dd>{tooltip.cell.roi.toUpperCase()}</dd></div>
            <div>
              <dt>Model</dt>
              <dd>{tooltip.cell.model} <span className="model-tag">{inferModelCategory(tooltip.cell.model)}</span></dd>
            </div>
            <div>
              <dt>Alignment</dt>
              <dd>
                {formatScore(tooltip.cell.score)}
              </dd>
            </div>
            <div><dt>ROI rank</dt><dd>{tooltip.cell.rankWithinRoi === null ? 'N/A' : `#${tooltip.cell.rankWithinRoi}`}</dd></div>
            <div><dt>Overall alignment</dt><dd>{formatScore(tooltip.cell.overallScore)}</dd></div>
          </dl>
        </div>,
        document.body,
      )}
    </section>
  );
}
