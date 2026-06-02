import { useState } from 'react';
import { ImageGrid } from './ImageGrid';
import type { ClusterPoint, ClusterSummary, ClusterView, EvidenceImage, GroundTruthPatient, ImageCategory } from '../types/data';

type ClusteringPanelProps = {
  voxelView: ClusterView | null;
  voxelStatus: 'idle' | 'loading' | 'error';
  visualView: ClusterView | null;
  imageCategories: Map<string, ImageCategory>;
  groundTruthPatients: GroundTruthPatient[];
  onOpenImage: (image: EvidenceImage) => void;
};

// Category highlight colors — must not clash with CLUSTER_COLORS (no red, blue, green, amber, violet, steel-blue, brown, teal)
const CATEGORY_COLORS: Record<ImageCategory, { fill: string; stroke: string }> = {
  'Faces':     { fill: '#F97316', stroke: '#7C2D12' }, // vivid orange
  'Places':    { fill: '#06B6D4', stroke: '#164E63' }, // cyan
  'Body Part': { fill: '#EC4899', stroke: '#831843' }, // hot pink
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
  view, onOpenImage, searchQuery, categoryHighlight, imageCategories,
}: {
  view: ClusterView;
  onOpenImage: (i: EvidenceImage) => void;
  searchQuery: string;
  categoryHighlight: ImageCategory | null;
  imageCategories: Map<string, ImageCategory>;
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
  const hasCategoryHighlight = categoryHighlight !== null;

  function isSearchMatch(name: string) { return hasSearch && name.toLowerCase().includes(needle); }
  function isCategoryMatch(name: string) { return hasCategoryHighlight && imageCategories.get(name) === categoryHighlight; }

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
            const searchMatched = isSearchMatch(p.imageName);
            const catMatched = isCategoryMatch(p.imageName);
            const anyFilter = hasSearch || hasCategoryHighlight;
            const highlighted = searchMatched || catMatched;
            const dimmed = anyFilter && !highlighted;

            let fill = baseColor;
            let stroke = 'var(--surface-raised)';
            let strokeWidth = 1;
            let r = isHovered ? 5.5 : 3.8;

            if (searchMatched) {
              fill = '#F59E0B';
              stroke = '#92400E';
              strokeWidth = 1.5;
              r = 7;
            } else if (catMatched && categoryHighlight) {
              const colors = CATEGORY_COLORS[categoryHighlight];
              fill = colors.fill;
              stroke = colors.stroke;
              strokeWidth = 1.5;
              r = isHovered ? 7 : 6;
            }

            return (
              <circle
                key={`${p.imageName}-${p.clusterLabel}`}
                cx={cx} cy={cy} r={r}
                fill={fill}
                opacity={dimmed ? 0.12 : 0.85}
                stroke={stroke}
                strokeWidth={strokeWidth}
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

const CATEGORIES: ImageCategory[] = ['Faces', 'Places', 'Body Part'];

const CATEGORY_LABELS: Record<ImageCategory, string> = {
  'Faces': 'Faces',
  'Places': 'Places',
  'Body Part': 'Body parts',
};

function ClusterSection({
  title, view, emptyMessage, onOpenImage, imageCategories, compact,
}: {
  title: string;
  view: ClusterView | null;
  emptyMessage: string;
  onOpenImage: (i: EvidenceImage) => void;
  imageCategories: Map<string, ImageCategory>;
  compact?: boolean;
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryHighlight, setCategoryHighlight] = useState<ImageCategory | null>(null);

  if (!view) {
    return compact ? (
      <p className="cluster-empty-note">{emptyMessage}</p>
    ) : (
      <section className="cluster-section">
        {title && <h3>{title}</h3>}
        <p className="cluster-empty-note">{emptyMessage}</p>
      </section>
    );
  }

  const needle = searchQuery.trim().toLowerCase();
  const matchCount = needle ? view.points.filter((p) => p.imageName.toLowerCase().includes(needle)).length : 0;

  function toggleCategory(cat: ImageCategory) {
    setCategoryHighlight((prev) => (prev === cat ? null : cat));
  }

  const inner = (
    <>
      {!compact && title && (
        <div className="cluster-section-heading">
          <div>
            <h3>{title}</h3>
            <p>{view.groups.length} clusters · {view.points.length} images</p>
          </div>
        </div>
      )}
      {compact && (
        <p className="cluster-section-meta">{view.groups.length} clusters · {view.points.length} images</p>
      )}
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

      <div className="scatter-category-row" role="group" aria-label="Highlight by image category">
        <span className="scatter-category-label">Highlight:</span>
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            type="button"
            className={`scatter-category-btn scatter-category-btn--${cat.toLowerCase().replace(' ', '-')} ${categoryHighlight === cat ? 'active' : ''}`}
            onClick={() => toggleCategory(cat)}
            aria-pressed={categoryHighlight === cat}
          >
            {CATEGORY_LABELS[cat]}
          </button>
        ))}
      </div>

      <ClusterScatter
        view={view}
        onOpenImage={onOpenImage}
        searchQuery={searchQuery}
        categoryHighlight={categoryHighlight}
        imageCategories={imageCategories}
      />
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
    </>
  );

  return compact ? <>{inner}</> : <section className="cluster-section">{inner}</section>;
}

const ROI_LABELS: Record<string, string> = {
  lffa: 'L FFA', rffa: 'R FFA',
  leba: 'L EBA', reba: 'R EBA',
  lppa: 'L PPA', rppa: 'R PPA',
};

function GroundTruthSection({
  patients, imageCategories, onOpenImage,
}: {
  patients: GroundTruthPatient[];
  imageCategories: Map<string, ImageCategory>;
  onOpenImage: (i: EvidenceImage) => void;
}) {
  const [activePatient, setActivePatient] = useState(patients[0]?.id ?? '');
  const [activeRoi, setActiveRoi] = useState<string | null>(null);

  if (patients.length === 0) {
    return (
      <section className="cluster-section">
        <h3>Ground truth clustering</h3>
        <p className="cluster-empty-note">No ground truth cluster data found.</p>
      </section>
    );
  }

  const patient = patients.find((p) => p.id === activePatient) ?? patients[0];
  const selectedRoi = activeRoi ?? patient.rois[0]?.roi ?? null;
  const roiView = patient.rois.find((r) => r.roi === selectedRoi);

  return (
    <section className="cluster-section">
      <h3>Ground truth clustering</h3>

      {/* Patient tabs */}
      <div className="gt-patient-tabs" role="tablist" aria-label="Select patient">
        {patients.map((p) => (
          <button
            key={p.id}
            type="button"
            role="tab"
            aria-selected={p.id === patient.id}
            className={`gt-patient-tab ${p.id === patient.id ? 'active' : ''}`}
            onClick={() => { setActivePatient(p.id); setActiveRoi(null); }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* ROI selector */}
      <div className="gt-roi-tabs" role="tablist" aria-label="Select ROI">
        {patient.rois.map(({ roi }) => (
          <button
            key={roi}
            type="button"
            role="tab"
            aria-selected={roi === selectedRoi}
            className={`gt-roi-tab ${roi === selectedRoi ? 'active' : ''}`}
            onClick={() => setActiveRoi(roi)}
          >
            {ROI_LABELS[roi] ?? roi.toUpperCase()}
          </button>
        ))}
      </div>

      {roiView ? (
        <ClusterSection
          title=""
          view={roiView.view}
          emptyMessage="No data for this ROI."
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          compact
        />
      ) : (
        <p className="cluster-empty-note">Select a ROI above.</p>
      )}
    </section>
  );
}

export function ClusteringPanel({ voxelView, voxelStatus, visualView, imageCategories, groundTruthPatients, onOpenImage }: ClusteringPanelProps) {
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
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
        />
      )}
      <ClusterSection
        title="Visual clustering"
        view={visualView}
        emptyMessage="No visual clustering output found."
        imageCategories={imageCategories}
        onOpenImage={onOpenImage}
      />
      <GroundTruthSection
        patients={groundTruthPatients}
        imageCategories={imageCategories}
        onOpenImage={onOpenImage}
      />
    </div>
  );
}
