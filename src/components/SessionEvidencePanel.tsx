import { useId, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';
import { ImageGrid } from './ImageGrid';
import type { EvidenceImage } from '../types/data';
import { UPLOADED_STIMULUS_MARKER_COLOR, type SessionEvidenceView } from '../utils/sessionEvidence';

type SessionEvidencePanelProps = {
  view: SessionEvidenceView;
  onOpenImage: (image: EvidenceImage) => void;
  onFocusStimulus: (stimulusId: string) => void;
};

type DistributionPoint = {
  image: EvidenceImage;
  origin: 'reference' | 'uploaded';
  px: number;
  outsideReferenceDomain: boolean;
};

type ActiveDistributionPoint = {
  index: number;
  trigger: 'pointer' | 'keyboard';
};

function ResponseDistribution({
  view,
  onOpenImage,
}: Pick<SessionEvidencePanelProps, 'view' | 'onOpenImage'>) {
  const width = 960;
  const height = 118;
  const left = 28;
  const right = 22;
  const axisY = 73;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const tooltipId = useId();
  const [active, setActive] = useState<ActiveDistributionPoint | null>(null);
  const [min, max] = view.referenceDomain;
  const span = max - min || 1;
  const x = (value: number) => left + ((value - min) / span) * (width - left - right);
  const pointFor = (image: EvidenceImage, origin: DistributionPoint['origin']): DistributionPoint => {
    const rawPx = x(image.value);
    const px = origin === 'uploaded' ? Math.max(left, Math.min(width - right, rawPx)) : rawPx;
    return { image, origin, px, outsideReferenceDomain: px !== rawPx };
  };
  const points = [
    ...view.referenceImages.map((image) => pointFor(image, 'reference')),
    ...view.uploadedImages.map((image) => pointFor(image, 'uploaded')),
  ];
  const navigationOrder = points
    .map((_, index) => index)
    .sort((a, b) => points[a].image.value - points[b].image.value || a - b);
  const focusedUploadIndex = points.findIndex((point) => point.image.stimulusId === view.focusedStimulus.id);
  const hasOutsideMarker = points.some((point) => point.outsideReferenceDomain);
  const activePoint = active && active.index >= 0 && active.index < points.length
    ? points[active.index]
    : undefined;
  const coincidentIndices = activePoint
    ? points.flatMap((point, index) => (Math.abs(point.px - activePoint.px) < 0.01 ? [index] : []))
    : [];
  const coincidentPosition = active ? coincidentIndices.indexOf(active.index) : -1;

  function nearestPointIndex(clientX: number, clientY?: number): number | null {
    const bounds = svgRef.current?.getBoundingClientRect();
    if (!bounds || bounds.width === 0 || points.length === 0) return null;
    const svgX = ((clientX - bounds.left) / bounds.width) * width;
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;

    points.forEach((point, index) => {
      const distance = Math.abs(point.px - svgX);
      const preferUploadAtSamePosition = distance === nearestDistance
        && point.origin === 'uploaded'
        && points[nearestIndex].origin !== 'uploaded';
      if (distance < nearestDistance || preferUploadAtSamePosition) {
        nearestIndex = index;
        nearestDistance = distance;
      }
    });

    const coincident = points.flatMap((point, index) => (
      Math.abs(point.px - points[nearestIndex].px) < 0.01 ? [index] : []
    ));
    if (clientY !== undefined && coincident.length > 1 && bounds.height > 0) {
      const svgY = ((clientY - bounds.top) / bounds.height) * height;
      const position = Math.max(0, Math.min(0.999, (svgY - (axisY - 14)) / 28));
      return coincident[Math.floor(position * coincident.length)];
    }

    return nearestIndex;
  }

  function handlePointerMove(event: PointerEvent<SVGRectElement>) {
    const index = nearestPointIndex(event.clientX, event.clientY);
    if (index !== null) {
      setActive((current) => (
        current?.index === index && current.trigger === 'pointer'
          ? current
          : { index, trigger: 'pointer' }
      ));
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (points.length === 0) return;
    const currentIndex = active?.index ?? (focusedUploadIndex >= 0 ? focusedUploadIndex : navigationOrder[0]);
    const orderIndex = Math.max(0, navigationOrder.indexOf(currentIndex));
    let nextIndex: number | null = null;

    if (event.key === 'ArrowLeft') {
      nextIndex = navigationOrder[Math.max(0, orderIndex - 1)];
    } else if (event.key === 'ArrowRight') {
      nextIndex = navigationOrder[Math.min(navigationOrder.length - 1, orderIndex + 1)];
    } else if (event.key === 'Home') {
      nextIndex = navigationOrder[0];
    } else if (event.key === 'End') {
      nextIndex = navigationOrder[navigationOrder.length - 1];
    } else if (event.key === 'Enter' || event.key === ' ') {
      if (!activePoint) return;
      event.preventDefault();
      onOpenImage(activePoint.image);
      return;
    } else if (event.key === 'Escape') {
      event.preventDefault();
      setActive(null);
      return;
    }

    if (nextIndex !== null) {
      event.preventDefault();
      setActive({ index: nextIndex, trigger: 'keyboard' });
    }
  }

  return (
    <figure className="session-distribution">
      <div
        className="session-distribution-plot"
        role="group"
        tabIndex={0}
        aria-label={`${view.roi.toUpperCase()} response number line with ${view.referenceImages.length} reference images and ${view.uploadedImages.length} uploaded ${view.uploadedImages.length === 1 ? 'stimulus' : 'stimuli'}. Use Left and Right Arrow keys to preview images and Enter to enlarge.`}
        aria-describedby={activePoint ? tooltipId : undefined}
        onFocus={() => {
          if (!active) {
            setActive({ index: focusedUploadIndex >= 0 ? focusedUploadIndex : navigationOrder[0], trigger: 'keyboard' });
          }
        }}
        onBlur={() => setActive(null)}
        onKeyDown={handleKeyDown}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          aria-hidden="true"
          focusable="false"
        >
          <line x1={left} y1={axisY} x2={width - right} y2={axisY} className="distribution-axis" />
          {points.map((point, index) => {
            const markerKey = `${point.origin}-${point.image.stimulusId ?? point.image.imageName}-${point.image.rank}`;
            const markerData = {
              'data-distribution-image': 'true',
              'data-image-name': point.image.imageName,
              'data-origin': point.origin,
              'data-response-value': point.image.value,
              'data-axis-y': axisY,
            };

            if (point.origin === 'reference') {
              return (
                <line
                  key={markerKey}
                  className="distribution-image-marker distribution-reference-tick"
                  x1={point.px}
                  x2={point.px}
                  y1={axisY - 4}
                  y2={axisY + 4}
                  {...markerData}
                >
                  <title>{`Reference: ${point.image.imageName}, ${point.image.value.toFixed(3)}`}</title>
                </line>
              );
            }

            return (
              <g
                key={markerKey}
                className="distribution-image-marker session-distribution-marker"
                transform={`translate(${point.px},${axisY})`}
                {...markerData}
              >
                <circle className="session-distribution-glow" r="7" fill={UPLOADED_STIMULUS_MARKER_COLOR} />
                <circle className="session-distribution-upload" r="4" fill={UPLOADED_STIMULUS_MARKER_COLOR} />
                <title>{`Uploaded: ${point.image.imageName}, ${point.image.value.toFixed(3)}${
                  point.outsideReferenceDomain ? '; outside the fixed reference range and shown at its boundary' : ''
                }`}</title>
              </g>
            );
          })}
          {activePoint && (
            <circle
              className={`distribution-active-marker distribution-active-marker--${activePoint.origin}`}
              cx={activePoint.px}
              cy={axisY}
              r="6"
              aria-hidden="true"
            />
          )}
          <rect
            className="distribution-interaction-band"
            x={left}
            y={axisY - 14}
            width={width - left - right}
            height={28}
            fill="transparent"
            data-distribution-interaction="nearest"
            onPointerEnter={handlePointerMove}
            onPointerMove={handlePointerMove}
            onPointerLeave={() => setActive((current) => current?.trigger === 'keyboard' ? current : null)}
            onClick={(event) => {
              const index = nearestPointIndex(event.clientX, event.clientY);
              if (index !== null) onOpenImage(points[index].image);
            }}
          />
          <text x={left} y={101} textAnchor="start">{min.toFixed(3)}</text>
          <text x={width - right} y={101} textAnchor="end">{max.toFixed(3)}</text>
        </svg>

        {activePoint && (
          <div
            id={tooltipId}
            role="tooltip"
            aria-live={active?.trigger === 'keyboard' ? 'polite' : undefined}
            className={`session-distribution-tooltip session-distribution-tooltip--${activePoint.origin}`}
            style={{
              left: `clamp(104px, ${(activePoint.px / width) * 100}%, calc(100% - 104px))`,
            } as CSSProperties}
          >
            <img src={activePoint.image.imageUrl} alt="" className="session-distribution-tooltip-image" />
            <p className="session-distribution-tooltip-name">{activePoint.image.imageName}</p>
            <p className="session-distribution-tooltip-meta">
              <span>
                {activePoint.origin === 'uploaded'
                  ? 'Uploaded'
                  : `Reference #${activePoint.image.referenceRank ?? activePoint.image.rank}`}
              </span>
              <span>Predicted response {activePoint.image.value.toFixed(3)}</span>
              {coincidentIndices.length > 1 && (
                <span>
                  {coincidentPosition + 1} of {coincidentIndices.length} images here · move up/down
                </span>
              )}
              {activePoint.outsideReferenceDomain && <span>Shown at the reference boundary</span>}
            </p>
          </div>
        )}
      </div>
      <figcaption>
        {view.referenceImages.length} Murty references + {view.uploadedImages.length}{' '}
        {view.uploadedImages.length === 1 ? 'upload' : 'uploads'} · NSD-1000 predicted mean.
        {' '}Hover / focus to preview · click to enlarge. Axis fixed to the reference range.
        {hasOutsideMarker && ' Out-of-range uploads appear at the boundary.'}
      </figcaption>
    </figure>
  );
}

function formatRank(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatReferenceInsertionRank(value: number): string {
  return value === 186 ? 'Below all 185' : `#${formatRank(value)}`;
}

function NeighborPair({ view, onOpenImage }: Pick<SessionEvidencePanelProps, 'view' | 'onOpenImage'>) {
  const images = [view.predecessor, view.successor].filter((image): image is EvidenceImage => image !== null);
  if (images.length === 0) return null;
  return <ImageGrid title="Lower / higher response" images={images} onOpenImage={onOpenImage} compact />;
}

export function SessionEvidencePanel({ view, onOpenImage, onFocusStimulus }: SessionEvidencePanelProps) {
  const { focusedResult, focusedStimulus } = view;
  const inclusion = focusedResult.entersTop
    ? 'Top 10'
    : focusedResult.entersBottom
      ? 'Bottom 10'
      : 'Neither';

  return (
    <div className="session-evidence-panel">
      <div className="session-evidence-heading">
        <div>
          <p className="eyebrow">Uploaded image</p>
          <h3>
            <span className="stimulus-dot" style={{ backgroundColor: focusedStimulus.color }} />
            {focusedStimulus.displayName}
          </h3>
        </div>
        <div className="session-focus-chips" aria-label="Focus uploaded stimulus">
          {view.uploadedImages.map((image) => (
            <button
              key={image.stimulusId}
              type="button"
              className={image.stimulusId === focusedStimulus.id ? 'active' : ''}
              onClick={() => image.stimulusId && onFocusStimulus(image.stimulusId)}
              style={{ borderColor: image.markerColor }}
            >
              {image.imageName}
            </button>
          ))}
        </div>
      </div>

      <dl className="summary-card-grid session-rank-summary">
        <div><dt>Predicted mean</dt><dd>{focusedResult.mean.toFixed(3)}</dd></div>
        <div>
          <dt title="Standard error across predicted voxel responses; not model uncertainty">Voxel SEM</dt>
          <dd>{focusedResult.sem === null ? '—' : focusedResult.sem.toFixed(3)}</dd>
        </div>
        <div>
          <dt title="Descending insertion rank among 185 reference predictions; ties receive their midpoint">Reference rank</dt>
          <dd>{formatReferenceInsertionRank(focusedResult.referenceRank)}</dd>
        </div>
        <div><dt>Reference percentile</dt><dd>{focusedResult.referencePercentile.toFixed(1)}th</dd></div>
        <div><dt title="Rank among reference images and uploads">Combined rank</dt><dd>#{focusedResult.sessionRank} / {focusedResult.sessionCount}</dd></div>
        <div><dt title="Membership in the combined top or bottom 10 reference images and uploads, independent of the display limit">Top / bottom 10</dt><dd>{inclusion}</dd></div>
      </dl>

      <ResponseDistribution view={view} onOpenImage={onOpenImage} />

      <div className="evidence-tier-grid">
        <ImageGrid title="Highest predicted responses" images={view.topImages} onOpenImage={onOpenImage} compact />
        <ImageGrid title="Lowest predicted responses" images={view.bottomImages} onOpenImage={onOpenImage} compact />
      </div>

      <details className="session-neighbors">
        <summary>Similar reference images</summary>
        <div className="session-neighbor-grid">
          <NeighborPair view={view} onOpenImage={onOpenImage} />
          <ImageGrid title="Closest by predicted mean" images={view.closestByMean} onOpenImage={onOpenImage} compact />
          <ImageGrid title="Similar response patterns" images={view.closestInVoxelSpace} onOpenImage={onOpenImage} compact />
        </div>
      </details>

      <details className="session-full-ranking">
        <summary>Full combined ranking ({view.mergedImages.length} images)</summary>
        <ImageGrid
          title={`Combined ${view.roi.toUpperCase()} ranking`}
          images={view.mergedImages}
          onOpenImage={onOpenImage}
          compact
        />
      </details>
    </div>
  );
}
