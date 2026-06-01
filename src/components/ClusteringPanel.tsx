import { useState } from 'react';
import { ImageGrid } from './ImageGrid';
import type { ClusterPoint, ClusterSummary, ClusterView, EvidenceImage } from '../types/data';

type ClusteringPanelProps = {
  voxelView: ClusterView | null;
  voxelStatus: 'idle' | 'loading' | 'error';
  visualView: ClusterView | null;
  onOpenImage: (image: EvidenceImage) => void;
};

// Muted, harmonious palette — reads well on the warm off-white surface
const CLUSTER_COLORS = [
  '#4F5FA6', // slate-indigo
  '#C0392B', // warm red
  '#2A7D5B', // forest green
  '#B45309', // amber
  '#6B5EA8', // soft violet
  '#1A7FA3', // steel blue
  '#8B5E3C', // warm brown
  '#3D7A6A', // teal
];

function fmtN(v: number | null | undefined) { return v == null ? '—' : v.toFixed(3); }
function fmtI(v: number | null | undefined) { return v == null ? '—' : String(v); }

function coords(point: ClusterPoint, mode: ClusterView['coordinateMode']): { x: number; y: number } | null {
  if (mode === 'plot' && point.plotX !== null && point.plotY !== null) return { x: point.plotX, y: point.plotY };
  if (mode === 'pca'  && point.pcaX  !== null && point.pcaY  !== null) return { x: point.pcaX,  y: point.pcaY  };
  if (mode === 'tsne' && point.tsneX !== null && point.tsneY !== null) return { x: point.tsneX, y: point.tsneY };
  return null;
}

type DotTooltip = {
  point: ClusterPoint;
  svgX: number;
  svgY: number;
};

function ClusterScatter({
  view, onOpenImage, searchQuery,
}: {
  view: ClusterView;
  onOpenImage: (i: EvidenceImage) => void;
  searchQuery: string;
}) {
  const [dotTooltip, setDotTooltip] = useState<DotTooltip | null>(null);

  if (!view.coordinateMode) {
    return <p className="cluster-empty-note">No projection coordinates available.</p>;
  }

  const pts = view.points
    .map((p) => { const c = coords(p, view.coordinateMode); return c ? { p, ...c } : null; })
    .filter((d): d is { p: ClusterPoint; x: number; y: number } => d !== null);

  if (pts.length === 0) return <p className="cluster-empty-note">No coordinate pairs found.</p>;

  const xs = pts.map((d) => d.x);
  const ys = pts.map((d) => d.y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const W = 520, H = 300, pad = 20;
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;

  const needle = searchQuery.trim().toLowerCase();
  const hasSearch = needle.length > 0;

  function isMatch(name: string) { return hasSearch && name.toLowerCase().includes(needle); }

  function handleDotClick(p: ClusterPoint) {
    onOpenImage({ imageName: p.imageName, imageUrl: p.imageUrl, value: p.value, rank: p.rank });
  }

  return (
    <figure className="cluster-scatter" onMouseLeave={() => setDotTooltip(null)}>
      <div style={{ position: 'relative' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${view.coordinateMode.toUpperCase()} scatter`}>
          <rect x="0" y="0" width={W} height={H} rx="10" />
          {pts.map(({ p, x, y }) => {
            const idx = Number(p.clusterLabel);
            const baseColor = CLUSTER_COLORS[Number.isFinite(idx) ? idx % CLUSTER_COLORS.length : 0];
            const cx = pad + ((x - minX) / xSpan) * (W - pad * 2);
            const cy = H - pad - ((y - minY) / ySpan) * (H - pad * 2);
            const isHovered = dotTooltip?.point.imageName === p.imageName;
            const matched = isMatch(p.imageName);
            const dimmed = hasSearch && !matched;
            return (
              <circle
                key={`${p.imageName}-${p.clusterLabel}`}
                cx={cx} cy={cy}
                r={matched ? 7 : isHovered ? 6 : 3.8}
                fill={matched ? '#F59E0B' : baseColor}
                opacity={dimmed ? 0.15 : 0.82}
                stroke={matched ? '#92400E' : 'var(--surface-raised)'}
                strokeWidth={matched ? 1.5 : 1}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setDotTooltip({ point: p, svgX: cx, svgY: cy })}
                onClick={() => handleDotClick(p)}
              />
            );
          })}
        </svg>

        {dotTooltip && (
          <div
            className="scatter-dot-tooltip"
            style={{
              left: `${(dotTooltip.svgX / W) * 100}%`,
              top: `${(dotTooltip.svgY / H) * 100}%`,
            }}
          >
            <img
              src={dotTooltip.point.imageUrl}
              alt={dotTooltip.point.imageName}
              className="scatter-dot-tooltip-img"
            />
            <p className="scatter-dot-tooltip-label">{dotTooltip.point.imageName}</p>
            <p className="scatter-dot-tooltip-sub">Cluster {dotTooltip.point.clusterLabel}</p>
          </div>
        )}
      </div>
      <figcaption>{view.coordinateMode.toUpperCase()} projection, coloured by cluster. Hover dots to preview · click to enlarge.</figcaption>
    </figure>
  );
}

function Metrics({ summary }: { summary: ClusterSummary | null }) {
  if (!summary) return null;
  return (
    <dl className="summary-card-grid cluster-summary-grid">
      <div><dt>Clusters</dt><dd>{fmtI(summary.bestK)}</dd></div>
      <div><dt>Silhouette</dt><dd>{fmtN(summary.silhouette)}</dd></div>
      <div><dt>Images</dt><dd>{fmtI(summary.nImages)}</dd></div>
      {summary.nVoxels !== undefined && <div><dt>Voxels</dt><dd>{fmtI(summary.nVoxels)}</dd></div>}
    </dl>
  );
}

function ClusterSection({
  title, view, emptyMessage, onOpenImage,
}: { title: string; view: ClusterView | null; emptyMessage: string; onOpenImage: (i: EvidenceImage) => void }) {
  const [searchQuery, setSearchQuery] = useState('');

  if (!view) {
    return (
      <section className="cluster-section">
        <h3>{title}</h3>
        <p className="cluster-empty-note">{emptyMessage}</p>
      </section>
    );
  }

  const needle = searchQuery.trim().toLowerCase();
  const matchCount = needle ? view.points.filter((p) => p.imageName.toLowerCase().includes(needle)).length : 0;

  return (
    <section className="cluster-section">
      <div className="cluster-section-heading">
        <div>
          <h3>{title}</h3>
          <p>{view.groups.length} clusters · {view.points.length} images</p>
        </div>
      </div>
      <Metrics summary={view.summary} />

      <div className="scatter-search-row">
        <input
          type="search"
          className="scatter-search-input"
          placeholder="Search image filename…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          aria-label="Search image filenames in scatter plot"
        />
        {needle && (
          <span className="scatter-search-count">
            {matchCount} match{matchCount !== 1 ? 'es' : ''}
          </span>
        )}
      </div>

      <ClusterScatter view={view} onOpenImage={onOpenImage} searchQuery={searchQuery} />
      <div className="cluster-group-list">
        {view.groups.map((g) => (
          <section className="cluster-group" key={`${title}-${g.label}`}>
            <div className="cluster-group-heading">
              <h4>Cluster {g.label}</h4>
              <span>{g.size} images</span>
            </div>
            <ImageGrid title={`Cluster ${g.label}`} images={g.images} onOpenImage={onOpenImage} compact />
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
          <p className="cluster-empty-note">Loading…</p>
        </section>
      ) : (
        <ClusterSection
          title="Voxel clustering"
          view={voxelStatus === 'error' ? null : voxelView}
          emptyMessage={
            voxelStatus === 'error'
              ? 'Cluster CSV could not be loaded.'
              : 'No voxel clustering found for this model and ROI.'
          }
          onOpenImage={onOpenImage}
        />
      )}
      <ClusterSection
        title="Visual clustering"
        view={visualView}
        emptyMessage="No visual clustering output found."
        onOpenImage={onOpenImage}
      />
    </div>
  );
}
