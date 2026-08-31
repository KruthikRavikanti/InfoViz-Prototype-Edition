import { useState } from 'react';
import { ImageGrid } from './ImageGrid';
import type { ClusterPoint, ClusterSummary, ClusterView, ClusteringMethod, EvidenceImage, GroundTruthPatient, ImageCategory } from '../types/data';
import type { CortexComparisonSession, SessionClusterOverlay } from '../types/session';
import { buildOptionalSessionOverlays, buildVoxelSessionOverlays, isCanonicalRoi } from '../utils/sessionEvidence';

type ClusteringPanelProps = {
  voxelView: ClusterView | null;
  voxelStatus: 'idle' | 'loading' | 'error';
  clusteringMethod: ClusteringMethod;
  clusteringMethodOptions: Array<{ id: ClusteringMethod; label: string; description: string }>;
  onClusteringMethodChange: (method: ClusteringMethod) => void;
  visualView: ClusterView | null;
  dreamsimView: ClusterView | null;
  imageCategories: Map<string, ImageCategory>;
  groundTruthPatients: GroundTruthPatient[];
  onOpenImage: (image: EvidenceImage) => void;
  session: CortexComparisonSession | null;
  selectedRoi: string;
  focusedStimulusId: string | null;
};

type ClusterSectionVariant = 'full' | 'board';

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
  view, onOpenImage, searchQuery, categoryHighlight, imageCategories, overlays = [], variant = 'full',
}: {
  view: ClusterView;
  onOpenImage: (i: EvidenceImage) => void;
  searchQuery: string;
  categoryHighlight: ImageCategory | null;
  imageCategories: Map<string, ImageCategory>;
  overlays?: SessionClusterOverlay[];
  variant?: ClusterSectionVariant;
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
  const W = 520;
  const H = variant === 'board' ? 210 : 300;
  const pad = variant === 'board' ? 16 : 20;
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  const hasOutsideOverlay = overlays.some((overlay) => (
    overlay.x < minX || overlay.x > maxX || overlay.y < minY || overlay.y > maxY
  ));

  const needle = searchQuery.trim().toLowerCase();
  const hasSearch = needle.length > 0;
  const hasCategoryHighlight = categoryHighlight !== null;

  function isSearchMatch(name: string) { return hasSearch && name.toLowerCase().includes(needle); }
  function isCategoryMatch(name: string) { return hasCategoryHighlight && imageCategories.get(name) === categoryHighlight; }

  function handleDotClick(p: ClusterPoint) {
    onOpenImage({ imageName: p.imageName, imageUrl: p.imageUrl, value: p.value, rank: p.rank });
  }

  function handleOverlayClick(overlay: SessionClusterOverlay) {
    onOpenImage({
      imageName: overlay.displayName,
      imageUrl: overlay.imageUrl,
      value: overlay.distance,
      valueLabel: `Uploaded · cluster ${overlay.clusterId}`,
      rank: 0,
      origin: 'uploaded',
      stimulusId: overlay.stimulusId,
      markerColor: overlay.color,
    });
  }

  return (
    <figure className={`cluster-scatter ${variant === 'board' ? 'cluster-scatter--board' : ''}`} onMouseLeave={() => setDotTooltip(null)}>
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
          {overlays.map((overlay) => {
            const rawCx = pad + ((overlay.x - minX) / xSpan) * (W - pad * 2);
            const rawCy = H - pad - ((overlay.y - minY) / ySpan) * (H - pad * 2);
            const cx = Math.max(pad, Math.min(W - pad, rawCx));
            const cy = Math.max(pad, Math.min(H - pad, rawCy));
            const outsideDomain = cx !== rawCx || cy !== rawCy;
            return (
              <g
                key={`session-${overlay.stimulusId}`}
                className="session-cluster-overlay"
                transform={`translate(${cx},${cy})`}
                role="button"
                tabIndex={0}
                data-stimulus-id={overlay.stimulusId}
                onClick={() => handleOverlayClick(overlay)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') handleOverlayClick(overlay);
                }}
              >
                <path d="M 0 -9 L 9 0 L 0 9 L -9 0 Z" fill={overlay.color} stroke="#fff" strokeWidth="2.4" />
                <circle r="10.5" fill="none" stroke={overlay.color} strokeWidth="1.5" />
                <title>
                  Uploaded: {overlay.displayName}; cluster {overlay.clusterId}; centroid distance {overlay.distance.toFixed(3)}
                  {outsideDomain ? '; projected outside the fixed reference domain and shown at its boundary' : ''}
                </title>
              </g>
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
      <figcaption>
        {view.coordinateMode.toUpperCase()} projection, coloured by cluster. Historical axes use only the fixed 185 references.
        {overlays.length > 0 ? ' Diamond markers are uploaded stimuli.' : ''} Hover dots to preview · click to enlarge.
        {hasOutsideOverlay ? ' Uploads outside the fixed reference range are pinned to the plot boundary.' : ''}
      </figcaption>
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
      {summary.nVoxels != null && <div><dt>Voxels</dt><dd>{fmtI(summary.nVoxels)}</dd></div>}
      {summary.nFeatures != null && <div><dt>Features</dt><dd>{fmtI(summary.nFeatures)}</dd></div>}
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
  title, view, emptyMessage, onOpenImage, imageCategories, overlays = [], compact, variant = 'full',
}: {
  title: string;
  view: ClusterView | null;
  emptyMessage: string;
  onOpenImage: (i: EvidenceImage) => void;
  imageCategories: Map<string, ImageCategory>;
  overlays?: SessionClusterOverlay[];
  compact?: boolean;
  variant?: ClusterSectionVariant;
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryHighlight, setCategoryHighlight] = useState<ImageCategory | null>(null);
  const isBoard = variant === 'board';

  if (!view) {
    return compact ? (
      <p className="cluster-empty-note">{emptyMessage}</p>
    ) : (
      <section className={`cluster-section ${isBoard ? 'cluster-section--board' : ''}`}>
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

      {view.summary?.dendrogramPng && (
        isBoard ? (
          <details className="cluster-dendrogram-details">
            <summary>Dendrogram</summary>
            <figure className="cluster-dendrogram">
              <img src={view.summary.dendrogramPng} alt={`${title || 'Selected clustering'} dendrogram`} loading="lazy" />
              <figcaption>
                {view.summary.distanceMetric ?? 'correlation'} distance
                {view.summary.linkageMethod ? ` · ${view.summary.linkageMethod} linkage` : ''}
              </figcaption>
            </figure>
          </details>
        ) : (
          <figure className="cluster-dendrogram">
            <img src={view.summary.dendrogramPng} alt={`${title || 'Selected clustering'} dendrogram`} loading="lazy" />
            <figcaption>
              Hierarchical dendrogram using {view.summary.distanceMetric ?? 'correlation'} distance
              {view.summary.linkageMethod ? ` and ${view.summary.linkageMethod} linkage` : ''}.
            </figcaption>
          </figure>
        )
      )}

      <div className="scatter-search-row">
        <input
          type="search"
          className="scatter-search-input"
          placeholder="Search image filename..."
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
        overlays={overlays}
        variant={variant}
      />
      {isBoard ? (
        <div className="cluster-size-list">
          {view.groups.map((g) => {
            const representative = g.images[0];
            return representative ? (
              <button
                type="button"
                className="cluster-size-chip"
                key={`${title}-${g.label}`}
                onClick={() => onOpenImage(representative)}
              >
                <span>Cluster {g.label}</span>
                <strong>{g.size}</strong>
              </button>
            ) : (
              <span className="cluster-size-chip" key={`${title}-${g.label}`}>
                <span>Cluster {g.label}</span>
                <strong>{g.size}</strong>
              </span>
            );
          })}
        </div>
      ) : (
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
      )}
    </>
  );

  return compact ? <>{inner}</> : <section className={`cluster-section ${isBoard ? 'cluster-section--board' : ''}`}>{inner}</section>;
}

const ROI_LABELS: Record<string, string> = {
  lffa: 'L FFA', rffa: 'R FFA',
  leba: 'L EBA', reba: 'R EBA',
  lppa: 'L PPA', rppa: 'R PPA',
};

function GroundTruthSection({
  patients, imageCategories, onOpenImage, sessionActive = false, variant = 'full',
}: {
  patients: GroundTruthPatient[];
  imageCategories: Map<string, ImageCategory>;
  onOpenImage: (i: EvidenceImage) => void;
  sessionActive?: boolean;
  variant?: ClusterSectionVariant;
}) {
  const [activePatient, setActivePatient] = useState(patients[0]?.id ?? '');
  const [activeRoi, setActiveRoi] = useState<string | null>(null);

  if (patients.length === 0) {
    return (
      <section className={`cluster-section ${variant === 'board' ? 'cluster-section--board' : ''}`}>
        <h3>Ground truth clustering</h3>
        {sessionActive && (
          <p className="ground-truth-session-note" data-no-upload-overlay="true">
            Uploaded stimuli are not shown in this measured fMRI view: they have predicted model responses, not ground-truth patient measurements.
          </p>
        )}
        <p className="cluster-empty-note">No ground truth cluster data found.</p>
      </section>
    );
  }

  const patient = patients.find((p) => p.id === activePatient) ?? patients[0];
  const selectedRoi = activeRoi ?? patient.rois[0]?.roi ?? null;
  const roiView = patient.rois.find((r) => r.roi === selectedRoi);

  return (
    <section className={`cluster-section ${variant === 'board' ? 'cluster-section--board' : ''}`}>
      <h3>Ground truth clustering</h3>
      {sessionActive && (
        <p className="ground-truth-session-note" data-no-upload-overlay="true">
          Uploaded stimuli are not shown in this measured fMRI view: they have predicted model responses, not ground-truth patient measurements.
        </p>
      )}

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
          variant={variant}
        />
      ) : (
        <p className="cluster-empty-note">Select a ROI above.</p>
      )}
    </section>
  );
}

function ClusteringMethodControl({
  value, options, onChange,
}: {
  value: ClusteringMethod;
  options: Array<{ id: ClusteringMethod; label: string; description: string }>;
  onChange: (method: ClusteringMethod) => void;
}) {
  return (
    <label className="cluster-method-control">
      Clustering method
      <select value={value} onChange={(event) => onChange(event.target.value as ClusteringMethod)}>
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label} · {option.description}
          </option>
        ))}
      </select>
    </label>
  );
}

function SessionVoxelAssignments({
  session,
  roi,
  method,
  focusedStimulusId,
}: {
  session: CortexComparisonSession;
  roi: 'ffa' | 'ppa' | 'eba';
  method: ClusteringMethod;
  focusedStimulusId: string | null;
}) {
  return (
    <section className="session-cluster-assignments">
      <h4>{method === 'kmeans_euclidean' ? 'Uploaded voxel assignments' : 'Inferred nearest-cluster assignments'}</h4>
      {method === 'hierarchical_correlation' && (
        <p>
          Hierarchical clustering has no out-of-sample insertion. These are labelled nearest-medoid assignments;
          the historical dendrogram was not rebuilt.
        </p>
      )}
      <div className="session-assignment-list">
        {session.roiResults[roi].stimuli.map((result) => {
          const stimulus = session.stimuli.find((candidate) => candidate.id === result.stimulusId)!;
          const hierarchical = result.hierarchicalAssignment;
          return (
            <article key={stimulus.id} className={stimulus.id === focusedStimulusId ? 'focused' : ''}>
              <div>
                <span className="stimulus-dot" style={{ backgroundColor: stimulus.color }} />
                <strong>{stimulus.displayName}</strong>
              </div>
              {method === 'kmeans_euclidean' ? (
                <>
                  <span>Cluster {result.voxelProjection.clusterId}</span>
                  <span>Centroid distance {result.voxelProjection.centroidDistance.toFixed(3)}</span>
                  <small>
                    Nearest references: {result.voxelProjection.nearestReferenceMembers.map((member) => member.imageName).join(', ') || 'none'}
                  </small>
                </>
              ) : (
                <>
                  <span>{hierarchical.method}</span>
                  <span>Cluster {hierarchical.clusterId} · distance {hierarchical.distance.toFixed(3)}</span>
                  <small>Medoid: {hierarchical.medoid.imageName}</small>
                </>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function ClusteringPanel({
  voxelView,
  voxelStatus,
  clusteringMethod,
  clusteringMethodOptions,
  onClusteringMethodChange,
  visualView,
  dreamsimView,
  imageCategories,
  groundTruthPatients,
  onOpenImage,
  session,
  selectedRoi,
  focusedStimulusId,
}: ClusteringPanelProps) {
  const roi = isCanonicalRoi(selectedRoi) ? selectedRoi : null;
  const voxelOverlays = session && roi && clusteringMethod === 'kmeans_euclidean'
    ? buildVoxelSessionOverlays(session, roi)
    : [];
  const visualOverlays = session && clusteringMethod === 'kmeans_euclidean'
    ? buildOptionalSessionOverlays(session, session.visualResults)
    : [];

  return (
    <div className="clustering-panel">
      <ClusteringMethodControl
        value={clusteringMethod}
        options={clusteringMethodOptions}
        onChange={onClusteringMethodChange}
      />
      <div className="cluster-board">
        {voxelStatus === 'loading' ? (
          <section className="cluster-section cluster-section--board">
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
            overlays={voxelOverlays}
            variant="board"
          />
        )}
        {session && roi && (
          <SessionVoxelAssignments
            session={session}
            roi={roi}
            method={clusteringMethod}
            focusedStimulusId={focusedStimulusId}
          />
        )}
        <ClusterSection
          title="Visual clustering"
          view={visualView}
          emptyMessage="No visual clustering output found."
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          overlays={visualOverlays}
          variant="board"
        />
        {session && visualOverlays.length === 0 && (
          <p className="session-cluster-note">
            Uploaded visual-feature placement is unavailable for this session; the fixed historical visual clusters remain unchanged.
          </p>
        )}
        <ClusterSection
          title="DreamSim embedding clustering"
          view={dreamsimView}
          emptyMessage="No DreamSim clustering output found."
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          variant="board"
        />
        {session && (
          <p className="session-cluster-note" data-dreamsim-overlay="disabled">
            Uploaded DreamSim placement is disabled: the archived reference bundle has 1,792 features, while the current
            Cortex embedding path produces 768. No incompatible projection is shown.
          </p>
        )}
        <GroundTruthSection
          patients={groundTruthPatients}
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          sessionActive={Boolean(session)}
          variant="board"
        />
      </div>
    </div>
  );
}
