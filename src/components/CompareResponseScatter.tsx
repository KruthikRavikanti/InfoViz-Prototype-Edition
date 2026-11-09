import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { SelectedHeatmapCell } from '../types/data';
import { responseAxisDomain, type SharedResponsePoint } from '../utils/responseComparison';
import { ComparisonPinMarker } from './ComparisonPinMarker';

export function comparisonModelLabel(model: string): string {
  return ({ clip_rn50: 'CLIP RN50', clip_rn101: 'CLIP RN101', clip_vit_b32: 'CLIP ViT-B/32' } as Record<string, string>)[model] ?? model;
}

type Props = {
  points: SharedResponsePoint[];
  cellA: SelectedHeatmapCell;
  cellB: SelectedHeatmapCell;
  pinnedImages: SharedResponsePoint[];
  onTogglePin: (imageName: string) => void;
};

const BLUE = '#2468A0';
const RED = '#C33C3C';
const W = 510;
const H = 500;
const LEFT = 72;
const TOP = 24;
const SIDE = 400;

export function CompareResponseScatter({ points, cellA, cellB, pinnedImages, onTogglePin }: Props) {
  type Preview = { point: SharedResponsePoint; x: number; y: number };
  const [hoverPreview, setHoverPreview] = useState<Preview | null>(null);
  const [focusPreview, setFocusPreview] = useState<Preview | null>(null);
  const preview = hoverPreview ?? focusPreview;
  const previewVisible = preview !== null;
  const tooltipId = useId();
  const figureRef = useRef<HTMLElement>(null);
  const [low, high] = responseAxisDomain(points);
  const scaleX = (value: number) => LEFT + (value - low) / (high - low) * SIDE;
  const scaleY = (value: number) => TOP + SIDE - (value - low) / (high - low) * SIDE;
  const aLabel = comparisonModelLabel(cellA.model);
  const bLabel = comparisonModelLabel(cellB.model);
  const pinByName = new Map(pinnedImages.map((image, index) => [image.imageName, index]));
  const ticks = Array.from({ length: Math.floor(high) - Math.ceil(low) + 1 }, (_, index) => Math.ceil(low) + index);

  useEffect(() => {
    if (!preview) return;
    const dismiss = () => { setHoverPreview(null); setFocusPreview(null); };
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
    };
  }, [previewVisible]);

  function showPreview(point: SharedResponsePoint, element: SVGElement, trigger: 'hover' | 'focus') {
    const rect = element.getBoundingClientRect();
    const width = 270;
    const height = 270;
    const next = { point,
      x: Math.max(8, Math.min(window.innerWidth - width - 8, rect.right + 14)),
      y: Math.max(8, Math.min(window.innerHeight - height - 8, rect.top - 90)),
    };
    if (trigger === 'focus') { setHoverPreview(null); setFocusPreview(next); }
    else setHoverPreview(next);
  }

  function moveFocus(index: number, delta: number) {
    const nodes = figureRef.current?.querySelectorAll<SVGGElement>('[data-response-image]');
    if (!nodes?.length) return;
    nodes[(index + delta + nodes.length) % nodes.length].focus();
  }

  return (
    <section className="compare-response-section" aria-label="Predicted response scatterplot">
      <div className="compare-response-intro">
        <h3>Relative predicted responses</h3>
        <span>{points.length} shared reference images</span>
      </div>
      <div className="compare-response-layout">
        <figure className="compare-response-figure" ref={figureRef}>
          <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`${aLabel} versus ${bLabel} standardized predicted mean responses`}>
            <rect x={LEFT} y={TOP} width={SIDE} height={SIDE} className="compare-response-frame" />
            {ticks.map(tick => <g key={tick} className="compare-response-axis">
              <line x1={scaleX(tick)} y1={TOP} x2={scaleX(tick)} y2={TOP + SIDE} />
              <line x1={LEFT} y1={scaleY(tick)} x2={LEFT + SIDE} y2={scaleY(tick)} />
              <text x={scaleX(tick)} y={TOP + SIDE + 20} textAnchor="middle">{tick}</text>
              <text x={LEFT - 12} y={scaleY(tick) + 4} textAnchor="end">{tick}</text>
            </g>)}
            <line className="compare-response-identity" x1={LEFT} y1={TOP + SIDE} x2={LEFT + SIDE} y2={TOP} />
            <text className="compare-response-identity-label" x={LEFT + SIDE - 12} y={TOP + 18} textAnchor="end">y = x</text>
            {/* Render highlighted points last so dense neighbors do not cover them. */}
            {[...points.filter(p => !p.gapDirection), ...points.filter(p => p.gapDirection)].map((point, index) => {
              const pinIndex = pinByName.get(point.imageName);
              const color = point.gapDirection === 'b' ? BLUE : point.gapDirection === 'a' ? RED : '#929292';
              return <g key={point.imageName}
                role="button" tabIndex={0}
                data-response-image={point.imageId}
                data-gap-direction={point.gapDirection ?? ''}
                data-z-a={point.zA} data-z-b={point.zB} data-z-gap={point.gap}
                data-rank-a={point.rankA} data-rank-b={point.rankB}
                aria-label={`${point.imageId}; ${aLabel} rank ${point.rankA}, z ${point.zA.toFixed(3)}; ${bLabel} rank ${point.rankB}, z ${point.zB.toFixed(3)}; ${pinIndex === undefined ? 'pin stimulus' : `unpin stimulus ${pinIndex + 1}`}`}
                aria-pressed={pinIndex !== undefined}
                aria-describedby={preview?.point.imageName === point.imageName ? tooltipId : undefined}
                onMouseEnter={event => showPreview(point, event.currentTarget, 'hover')}
                onMouseLeave={() => setHoverPreview(null)}
                onFocus={event => showPreview(point, event.currentTarget, 'focus')}
                onBlur={() => setFocusPreview(null)}
                onClick={() => onTogglePin(point.imageName)}
                onKeyDown={event => {
                  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onTogglePin(point.imageName); }
                  if (event.key === 'Escape') {
                    // Let Escape reach the comparison dialog from focused points too.
                    setHoverPreview(null); setFocusPreview(null);
                  }
                  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') { event.preventDefault(); moveFocus(index, 1); }
                  if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') { event.preventDefault(); moveFocus(index, -1); }
                }}>
                <circle className="compare-response-hit" cx={scaleX(point.zA)} cy={scaleY(point.zB)} r={7} />
                <circle className="compare-response-point" cx={scaleX(point.zA)} cy={scaleY(point.zB)}
                  r={point.gapDirection ? 5.4 : 3.8} fill={color} opacity={point.gapDirection ? 1 : .65} />
              </g>;
            })}
            {pinnedImages.map((point, index) => <ComparisonPinMarker key={point.imageName} index={index}
              x={scaleX(point.zA)} y={scaleY(point.zB)} label={`${index + 1} · ${point.imageId}`}
              labelSide={point.zA > (low + high) / 2 ? 'left' : 'right'} labelDy={index === 0 ? 21 : -16} />)}
            <text className="compare-response-axis-title" x={LEFT + SIDE / 2} y={H - 20} textAnchor="middle">{aLabel} · {cellA.roi.toUpperCase()} (z-score)</text>
            <text className="compare-response-axis-title" transform={`translate(19,${TOP + SIDE / 2}) rotate(-90)`} textAnchor="middle">{bLabel} · {cellB.roi.toUpperCase()} (z-score)</text>
          </svg>
          <figcaption>Each model standardized separately across shared references. Dashed line: equal z-scores.</figcaption>
        </figure>
        <div className="compare-gap-examples" aria-label="Largest standardized response gaps">
          {(['b', 'a'] as const).map(direction => <section key={direction} className={`compare-gap-group compare-gap-group--${direction}`}>
            <h4>Higher relative response in {direction === 'b' ? bLabel : aLabel}</h4>
            <div className="compare-gap-grid">
              {points.filter(p => p.gapDirection === direction).sort((a, b) => direction === 'b' ? b.gap - a.gap : a.gap - b.gap).map(point => <button
                type="button" className="compare-gap-example" key={point.imageName}
                onClick={() => onTogglePin(point.imageName)} aria-pressed={pinByName.has(point.imageName)}
                aria-label={`${pinByName.has(point.imageName) ? 'Unpin' : 'Pin'} ${point.imageId}`}>
                <img src={point.imageUrl} alt={point.imageId} />
                <strong>{point.imageId}</strong>
                <span>A #{point.rankA} · B #{point.rankB}</span>
                <span>Δz {point.gap >= 0 ? '+' : ''}{point.gap.toFixed(3)}</span>
              </button>)}
            </div>
          </section>)}
          <p>Color marks the two largest gaps in each direction. Pin shapes mark your selections.</p>
        </div>
      </div>
      {preview && createPortal(<div className="compare-response-tooltip" id={tooltipId} role="tooltip"
        style={{ left: preview.x, top: preview.y }}>
        <img src={preview.point.imageUrl} alt="" />
        <strong>{preview.point.imageId}</strong>
        <table><thead><tr><th>Model</th><th>Rank</th><th>z-score</th></tr></thead><tbody>
          <tr><th>{aLabel}</th><td>#{preview.point.rankA}</td><td>{preview.point.zA.toFixed(3)}</td></tr>
          <tr><th>{bLabel}</th><td>#{preview.point.rankB}</td><td>{preview.point.zB.toFixed(3)}</td></tr>
        </tbody></table>
        <span>Click or press Enter to {pinByName.has(preview.point.imageName) ? 'unpin' : 'pin'}.</span>
      </div>, document.body)}
    </section>
  );
}
