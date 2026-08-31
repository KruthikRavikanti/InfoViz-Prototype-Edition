import { ImageGrid } from './ImageGrid';
import type { EvidenceImage } from '../types/data';
import type { SessionEvidenceView } from '../utils/sessionEvidence';

type SessionEvidencePanelProps = {
  view: SessionEvidenceView;
  onOpenImage: (image: EvidenceImage) => void;
  onFocusStimulus: (stimulusId: string) => void;
};

function ResponseDistribution({ view }: { view: SessionEvidenceView }) {
  const width = 640;
  const height = 118;
  const left = 28;
  const right = 22;
  const axisY = 73;
  const [min, max] = view.referenceDomain;
  const span = max - min || 1;
  const hasOutsideMarker = view.uploadedImages.some((image) => image.value < min || image.value > max);
  const x = (value: number) => left + ((value - min) / span) * (width - left - right);

  return (
    <figure className="session-distribution">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${view.roi.toUpperCase()} historical response distribution with uploaded stimuli`}>
        <line x1={left} y1={axisY} x2={width - right} y2={axisY} className="distribution-axis" />
        {view.referenceImages.map((image, index) => (
          <circle
            key={image.imageName}
            cx={x(image.value)}
            cy={axisY - 3 - (index % 5) * 4}
            r="2.2"
            className="distribution-reference"
          >
            <title>{image.imageName}: {image.value.toFixed(3)}</title>
          </circle>
        ))}
        {view.uploadedImages.map((image, index) => {
          const rawPx = x(image.value);
          const px = Math.max(left, Math.min(width - right, rawPx));
          const outsideReferenceDomain = px !== rawPx;
          const color = image.markerColor ?? '#111827';
          return (
            <g key={image.stimulusId ?? image.imageName} transform={`translate(${px},${axisY})`}>
              <path d="M 0 -14 L 8 -4 L 0 6 L -8 -4 Z" fill={color} stroke="#fff" strokeWidth="2" />
              <text x="0" y={-21 - (index % 2) * 12} textAnchor="middle" fill={color}>{image.imageName}</text>
              <title>
                Uploaded: {image.imageName}, {image.value.toFixed(3)}
                {outsideReferenceDomain ? '; outside the fixed historical range and shown at its boundary' : ''}
              </title>
            </g>
          );
        })}
        <text x={left} y={101} textAnchor="start">{min.toFixed(3)}</text>
        <text x={width - right} y={101} textAnchor="end">{max.toFixed(3)}</text>
      </svg>
      <figcaption>
        Predicted response under NSD-1000 mapping. The axis domain is fixed to the 185 historical reference images.
        {hasOutsideMarker && ' Markers outside the fixed historical range are pinned to the boundary.'}
      </figcaption>
    </figure>
  );
}

function formatRank(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatReferenceInsertionRank(value: number): string {
  return value === 186 ? 'After all 185 references' : `#${formatRank(value)} relative to 185 references`;
}

function NeighborPair({ view, onOpenImage }: Pick<SessionEvidencePanelProps, 'view' | 'onOpenImage'>) {
  const images = [view.predecessor, view.successor].filter((image): image is EvidenceImage => image !== null);
  if (images.length === 0) return null;
  return <ImageGrid title="Closest lower / higher references" images={images} onOpenImage={onOpenImage} compact />;
}

export function SessionEvidencePanel({ view, onOpenImage, onFocusStimulus }: SessionEvidencePanelProps) {
  const { focusedResult, focusedStimulus } = view;
  const inclusion = focusedResult.entersTop
    ? 'Enters the displayed top set'
    : focusedResult.entersBottom
      ? 'Enters the displayed bottom set'
      : 'Outside the displayed top/bottom sets';

  return (
    <div className="session-evidence-panel">
      <div className="session-evidence-heading">
        <div>
          <p className="eyebrow">Uploaded stimulus context</p>
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
        <div><dt>SEM</dt><dd>{focusedResult.sem === null ? '—' : focusedResult.sem.toFixed(3)}</dd></div>
        <div><dt>Reference insertion rank</dt><dd>{formatReferenceInsertionRank(focusedResult.referenceRank)}</dd></div>
        <div><dt>Reference percentile</dt><dd>{focusedResult.referencePercentile.toFixed(1)}th</dd></div>
        <div><dt>Combined rank</dt><dd>#{focusedResult.sessionRank} / {focusedResult.sessionCount}</dd></div>
        <div><dt>Top / bottom</dt><dd>{inclusion}</dd></div>
      </dl>

      <ResponseDistribution view={view} />

      <div className="evidence-tier-grid">
        <ImageGrid title="Session top images" images={view.topImages} onOpenImage={onOpenImage} compact />
        <ImageGrid title="Session bottom images" images={view.bottomImages} onOpenImage={onOpenImage} compact />
      </div>

      <div className="session-neighbor-grid">
        <NeighborPair view={view} onOpenImage={onOpenImage} />
        <ImageGrid title="Closest by predicted mean" images={view.closestByMean} onOpenImage={onOpenImage} compact />
        <ImageGrid title="Closest in voxel space" images={view.closestInVoxelSpace} onOpenImage={onOpenImage} compact />
      </div>

      <details className="session-full-ranking">
        <summary>Show full combined ranking ({view.mergedImages.length} images)</summary>
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
