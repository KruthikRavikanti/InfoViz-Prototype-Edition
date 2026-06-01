import { ImageGrid } from './ImageGrid';
import type { ClusterPoint, ClusterSummary, ClusterView, EvidenceImage } from '../types/data';

type ClusteringPanelProps = {
  voxelView: ClusterView | null;
  voxelStatus: 'idle' | 'loading' | 'error';
  visualView: ClusterView | null;
  onOpenImage: (image: EvidenceImage) => void;
};

// Distinct, vibrant but not garish colors for dark backgrounds
const clusterColors = [
  '#7c8cff', // accent blue
  '#ff8a5b', // compare orange
  '#5fd0a5', // success teal
  '#f2c66d', // warning yellow
  '#a78bfa', // soft violet
  '#38bdf8', // sky blue
  '#fb7185', // soft red
  '#34d399', // emerald
];

function formatNumber(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : value.toFixed(3);
}

function formatInteger(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : String(value);
}

function getPointCoordinates(point: ClusterPoint, mode: ClusterView['coordinateMode']): { x: number; y: number } | null {
  if (mode === 'plot' && point.plotX !== null && point.plotY !== null) {
    return { x: point.plotX, y: point.plotY };
  }

  if (mode === 'pca' && point.pcaX !== null && point.pcaY !== null) {
    return { x: point.pcaX, y: point.pcaY };
  }

  if (mode === 'tsne' && point.tsneX !== null && point.tsneY !== null) {
    return { x: point.tsneX, y: point.tsneY };
  }

  return null;
}

function ClusterScatter({ view }: { view: ClusterView }) {
  if (!view.coordinateMode) {
    return <p className="cluster-empty-note">No projection coordinates found for this clustering output.</p>;
  }

  const drawablePoints = view.points
    .map((point) => {
      const coordinates = getPointCoordinates(point, view.coordinateMode);
      return coordinates ? { point, ...coordinates } : null;
    })
    .filter((point): point is { point: ClusterPoint; x: number; y: number } => point !== null);

  if (drawablePoints.length === 0) {
    return <p className="cluster-empty-note">No complete coordinate pairs found for this clustering output.</p>;
  }

  const xValues = drawablePoints.map((point) => point.x);
  const yValues = drawablePoints.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const width = 520;
  const height = 300;
  const padding = 20;
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;

  return (
    <figure className="cluster-scatter">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${view.coordinateMode.toUpperCase()} cluster scatter plot`}>
        <rect x="0" y="0" width={width} height={height} rx="10" />
        {drawablePoints.map(({ point, x, y }) => {
          const labelIndex = Number(point.clusterLabel);
          const color = clusterColors[Number.isFinite(labelIndex) ? labelIndex % clusterColors.length : 0];
          const cx = padding + ((x - minX) / xSpan) * (width - padding * 2);
          const cy = height - padding - ((y - minY) / ySpan) * (height - padding * 2);

          return (
            <circle key={`${point.imageName}-${point.clusterLabel}`} cx={cx} cy={cy} r="3.8" fill={color}>
              <title>{`${point.imageName} | cluster ${point.clusterLabel}`}</title>
            </circle>
          );
        })}
      </svg>
      <figcaption>{view.coordinateMode.toUpperCase()} projection, colored by cluster.</figcaption>
    </figure>
  );
}

function ClusterSummaryMetrics({ summary }: { summary: ClusterSummary | null }) {
  if (!summary) {
    return null;
  }

  return (
    <dl className="summary-card-grid cluster-summary-grid">
      <div>
        <dt>Clusters</dt>
        <dd>{formatInteger(summary.bestK)}</dd>
      </div>
      <div>
        <dt>Silhouette</dt>
        <dd>{formatNumber(summary.silhouette)}</dd>
      </div>
      <div>
        <dt>Images</dt>
        <dd>{formatInteger(summary.nImages)}</dd>
      </div>
      {summary.nVoxels !== undefined && (
        <div>
          <dt>Voxels</dt>
          <dd>{formatInteger(summary.nVoxels)}</dd>
        </div>
      )}
    </dl>
  );
}

function ClusterViewSection({
  title,
  view,
  emptyMessage,
  onOpenImage,
}: {
  title: string;
  view: ClusterView | null;
  emptyMessage: string;
  onOpenImage: (image: EvidenceImage) => void;
}) {
  if (!view) {
    return (
      <section className="cluster-section">
        <h3>{title}</h3>
        <p className="cluster-empty-note">{emptyMessage}</p>
      </section>
    );
  }

  return (
    <section className="cluster-section">
      <div className="cluster-section-heading">
        <div>
          <h3>{title}</h3>
          <p>
            {view.groups.length} clusters &middot; {view.points.length} images
          </p>
        </div>
      </div>

      <ClusterSummaryMetrics summary={view.summary} />
      <ClusterScatter view={view} />

      <div className="cluster-group-list">
        {view.groups.map((group) => (
          <section className="cluster-group" key={`${title}-${group.label}`}>
            <div className="cluster-group-heading">
              <h4>Cluster {group.label}</h4>
              <span>{group.size} images</span>
            </div>
            <ImageGrid title={`Cluster ${group.label} representatives`} images={group.images} onOpenImage={onOpenImage} compact />
          </section>
        ))}
      </div>
    </section>
  );
}

export function ClusteringPanel({ voxelView, voxelStatus, visualView, onOpenImage }: ClusteringPanelProps) {
  return (
    <div className="clustering-panel">
      {voxelStatus === 'loading' ? (
        <section className="cluster-section">
          <h3>Voxel clustering</h3>
          <p className="cluster-empty-note">Loading voxel clusters…</p>
        </section>
      ) : (
        <ClusterViewSection
          title="Voxel clustering"
          view={voxelStatus === 'error' ? null : voxelView}
          emptyMessage={
            voxelStatus === 'error'
              ? 'Voxel clustering exists for this cell, but the cluster CSV could not be loaded.'
              : 'No voxel clustering output found for this model and ROI.'
          }
          onOpenImage={onOpenImage}
        />
      )}

      <ClusterViewSection
        title="Visual clustering"
        view={visualView}
        emptyMessage="No visual clustering output found."
        onOpenImage={onOpenImage}
      />
    </div>
  );
}
