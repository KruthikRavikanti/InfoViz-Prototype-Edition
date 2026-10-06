import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { ImageGrid } from './ImageGrid';
import { VoxelSelectivityAtlasDialog } from './VoxelSelectivityAtlasDialog';
import { JointDendrogram } from './JointDendrogram';
import type { ClusterPoint, ClusterSummary, ClusterView, ClusteringMethod, EvidenceImage, GroundTruthPatient, ImageCategory } from '../types/data';
import type { CanonicalRoi, CortexComparisonSession, SessionClusterOverlay } from '../types/session';
import {
  buildOptionalSessionOverlays,
  buildVoxelSessionOverlays,
  isCanonicalRoi,
  UPLOADED_STIMULUS_MARKER_COLOR,
} from '../utils/sessionEvidence';
import { buildJointClusterView, jointClusterMedoidName } from '../utils/jointClustering';

export type VoxelRoiClusterState = {
  view: ClusterView | null;
  status: 'idle' | 'loading' | 'error';
};

type ClusteringPanelProps = {
  voxelRoiStates: Record<CanonicalRoi, VoxelRoiClusterState>;
  clusteringMethod: ClusteringMethod;
  clusteringMethodOptions: Array<{ id: ClusteringMethod; label: string; description: string }>;
  onClusteringMethodChange: (method: ClusteringMethod) => void;
  visualView: ClusterView | null;
  dreamsimView: ClusterView | null;
  imageCategories: Map<string, ImageCategory>;
  groundTruthPatients: GroundTruthPatient[];
  onOpenImage: (image: EvidenceImage) => void;
  session: CortexComparisonSession | null;
  selectedModel: string;
  selectedRoi: string;
  selectivityAtlasAvailable: boolean;
  showSelectivityLauncher?: boolean;
  focusedStimulusId: string | null;
};

type ClusterSectionVariant = 'full' | 'board';
type LinkedPlotItem = { kind: 'reference' | 'uploaded'; id: string; label: string };

const ROI_COMPARISON_ORDER: CanonicalRoi[] = ['ffa', 'eba', 'ppa'];

function jointClusteringNote(view: ClusterView | null, hasHierarchy: boolean): string | null {
  if (!hasHierarchy) {
    return 'This upload session predates joint clustering. Reopen Explore visualization in Virtual Cortex to rebuild it.';
  }
  if (view && !view.jointHierarchy) {
    return 'Joint clustering could not be matched to these reference images. Only the reference clusters are shown.';
  }
  if (view?.jointHierarchy && !view.uploadAxisAlignment) {
    return 'The joint tree is available, but upload markers are hidden because the reference axes could not be verified.';
  }
  return null;
}

function jointRoiStates(
  states: Record<CanonicalRoi, VoxelRoiClusterState>,
  method: ClusteringMethod,
  session: CortexComparisonSession | null,
): Record<CanonicalRoi, VoxelRoiClusterState> {
  if (method !== 'hierarchical_correlation' || !session) return states;
  return Object.fromEntries(ROI_COMPARISON_ORDER.map((roi) => {
    const state = states[roi];
    const hierarchy = session.roiResults[roi].jointHierarchy;
    const view = state.view && hierarchy && state.view.jointHierarchy !== hierarchy
      ? buildJointClusterView(state.view, hierarchy, session) ?? state.view
      : state.view;
    return [roi, { ...state, view }];
  })) as Record<CanonicalRoi, VoxelRoiClusterState>;
}

function keepFocusInsideDialog(event: KeyboardEvent<HTMLElement>) {
  if (event.key !== 'Tab') return;

  const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
    'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], summary, [tabindex]:not([tabindex="-1"])',
  )).filter((element) => element.getClientRects().length > 0);
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (!first || !last) return;

  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

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

function projectionLabel(view: ClusterView): string {
  return view.coordinateMode === 'plot' ? 'PCA' : view.coordinateMode?.toUpperCase() ?? 'Unavailable';
}

function representativeDistanceLabel(view: ClusterView): string {
  return view.summary?.distanceToClusterRepresentative?.includes('medoid')
    || view.summary?.clusteringMethod === 'hierarchical'
    ? 'Medoid distance'
    : 'Centroid distance';
}

function overlayDistanceLabel(overlay: SessionClusterOverlay): string {
  return overlay.assignmentMethod === 'nearest-centroid assignment' ? 'Centroid distance' : 'Medoid correlation distance';
}

function overlayAssignmentDescription(overlay: SessionClusterOverlay): string {
  if (overlay.assignmentMethod === 'joint hierarchical clustering') {
    return `joint hierarchical cluster ${overlay.clusterId}, correlation distance to cluster medoid ${overlay.distance.toFixed(3)}, medoid ${overlay.medoidImageName}`;
  }
  return overlay.assignmentMethod === 'nearest-medoid assignment'
    ? `nearest-medoid assignment to cluster ${overlay.clusterId}, correlation distance ${overlay.distance.toFixed(3)}, medoid ${overlay.medoidImageName}`
    : `cluster ${overlay.clusterId}, centroid distance ${overlay.distance.toFixed(3)}`;
}

function coords(point: ClusterPoint, mode: ClusterView['coordinateMode']): { x: number; y: number } | null {
  if (mode === 'plot' && point.plotX !== null && point.plotY !== null) return { x: point.plotX, y: point.plotY };
  if (mode === 'pca'  && point.pcaX  !== null && point.pcaY  !== null) return { x: point.pcaX,  y: point.pcaY  };
  if (mode === 'tsne' && point.tsneX !== null && point.tsneY !== null) return { x: point.tsneX, y: point.tsneY };
  return null;
}

type DotTooltip =
  | {
      kind: 'reference';
      trigger: 'hover';
      point: ClusterPoint;
      anchorX: number;
      anchorY: number;
    }
  | {
      kind: 'uploaded';
      trigger: 'hover' | 'focus';
      overlay: SessionClusterOverlay;
      anchorX: number;
      anchorY: number;
    };

function tooltipAnchor(target: SVGGraphicsElement): { anchorX: number; anchorY: number } {
  const bounds = target.getBoundingClientRect();
  return { anchorX: bounds.left + bounds.width / 2, anchorY: bounds.top + bounds.height / 2 };
}

function ClusterScatter({
  view,
  onOpenImage,
  searchQuery,
  categoryHighlight,
  imageCategories,
  overlays = [],
  variant = 'full',
  linkedItem = null,
  onLinkedItemChange,
  roi,
}: {
  view: ClusterView;
  onOpenImage: (i: EvidenceImage) => void;
  searchQuery: string;
  categoryHighlight: ImageCategory | null;
  imageCategories: Map<string, ImageCategory>;
  overlays?: SessionClusterOverlay[];
  variant?: ClusterSectionVariant;
  linkedItem?: LinkedPlotItem | null;
  onLinkedItemChange?: (item: LinkedPlotItem | null) => void;
  roi?: CanonicalRoi;
}) {
  const [dotTooltip, setDotTooltip] = useState<DotTooltip | null>(null);
  const tooltipId = useId();
  const tooltipRef = useRef<HTMLDivElement>(null);
  const tooltipVisible = dotTooltip !== null;

  useEffect(() => {
    setDotTooltip(null);
    onLinkedItemChange?.(null);
  }, [view, onLinkedItemChange]);

  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    if (!dotTooltip || !tooltip) return;
    const height = tooltip.offsetHeight;
    const above = dotTooltip.anchorY - height - 10;
    const preferredTop = above >= 8 ? above : dotTooltip.anchorY + 10;
    tooltip.style.top = `${Math.max(8, Math.min(window.innerHeight - height - 8, preferredTop))}px`;
  }, [dotTooltip]);

  useEffect(() => {
    if (!tooltipVisible) return;
    // Floating previews must not remain at a stale screen position after a scroll.
    const dismiss = () => {
      setDotTooltip(null);
      onLinkedItemChange?.(null);
    };
    window.addEventListener('scroll', dismiss, true);
    window.addEventListener('resize', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
    };
  }, [tooltipVisible, onLinkedItemChange]);

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
    onOpenImage({
      imageName: p.imageName,
      imageUrl: p.imageUrl,
      value: p.value,
      valueLabel: p.distanceToCentroid === null
        ? `Cluster ${p.clusterLabel}`
        : `${representativeDistanceLabel(view)} ${p.distanceToCentroid.toFixed(3)}`,
      rank: p.rank,
    });
  }

  function handleOverlayClick(overlay: SessionClusterOverlay) {
    setDotTooltip(null);
    onLinkedItemChange?.(null);
    onOpenImage({
      imageName: overlay.displayName,
      imageUrl: overlay.imageUrl,
      value: overlay.distance,
      valueLabel: overlay.assignmentMethod === 'joint hierarchical clustering'
        ? `Uploaded · joint hierarchical cluster ${overlay.clusterId} · medoid correlation distance ${overlay.distance.toFixed(3)}`
        : `Uploaded · cluster ${overlay.clusterId}`,
      rank: 0,
      origin: 'uploaded',
      stimulusId: overlay.stimulusId,
      markerColor: overlay.color,
    });
  }

  function clearReferenceHover() {
    setDotTooltip(null);
    onLinkedItemChange?.(null);
  }

  function clearUploadHover() {
    if (dotTooltip?.kind === 'uploaded' && dotTooltip.trigger === 'focus') return;
    setDotTooltip(null);
    onLinkedItemChange?.(null);
  }

  return (
    <figure
      className={`cluster-scatter ${variant === 'board' ? 'cluster-scatter--board' : ''}`}
      onMouseLeave={() => {
        if (dotTooltip?.kind === 'uploaded' && dotTooltip.trigger === 'focus') return;
        setDotTooltip(null);
        onLinkedItemChange?.(null);
      }}
    >
      <div style={{ position: 'relative' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`${roi ? `${roi.toUpperCase()} ` : ''}${projectionLabel(view)} image-cluster projection`}>
          <rect x="0" y="0" width={W} height={H} rx="10" />
          {pts.map(({ p, x, y }) => {
            const idx = Number(p.clusterLabel);
            const baseColor = CLUSTER_COLORS[Number.isFinite(idx) ? idx % CLUSTER_COLORS.length : 0];
            const cx = pad + ((x - minX) / xSpan) * (W - pad * 2);
            const cy = H - pad - ((y - minY) / ySpan) * (H - pad * 2);
            const isHovered = dotTooltip?.kind === 'reference' && dotTooltip.point.imageName === p.imageName;
            const isLinked = linkedItem?.kind === 'reference' && linkedItem.id === p.imageName;
            const searchMatched = isSearchMatch(p.imageName);
            const catMatched = isCategoryMatch(p.imageName);
            const anyFilter = hasSearch || hasCategoryHighlight;
            const highlighted = searchMatched || catMatched;
            const dimmed = anyFilter && !highlighted;

            let fill = baseColor;
            let stroke = 'var(--surface-raised)';
            let strokeWidth = 1;
            let r = isLinked ? 6.5 : isHovered ? 5.5 : 3.8;

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
                className={isLinked ? 'cluster-reference-point cluster-reference-point--linked' : 'cluster-reference-point'}
                cx={cx} cy={cy} r={r}
                fill={fill}
                style={{
                  cursor: 'pointer',
                  opacity: isLinked ? 1 : dimmed ? 0.12 : 0.85,
                  stroke: isLinked ? 'var(--accent)' : stroke,
                  strokeWidth: isLinked ? 2 : strokeWidth,
                  filter: isLinked ? 'drop-shadow(0 0 3px color-mix(in srgb, var(--accent) 38%, transparent))' : undefined,
                }}
                data-link-key={`reference:${p.imageName}`}
                data-image-name={p.imageName}
                data-roi={roi}
                data-linked-highlighted={isLinked ? 'true' : undefined}
                onMouseEnter={(event) => {
                  setDotTooltip({ kind: 'reference', trigger: 'hover', point: p, ...tooltipAnchor(event.currentTarget) });
                  onLinkedItemChange?.({ kind: 'reference', id: p.imageName, label: p.imageName });
                }}
                onMouseLeave={clearReferenceHover}
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
            const isLinked = linkedItem?.kind === 'uploaded' && linkedItem.id === overlay.stimulusId;
            return (
              <g
                key={`session-${overlay.stimulusId}`}
                className={`session-cluster-overlay ${isLinked ? 'session-cluster-overlay--linked' : ''}`}
                transform={`translate(${cx},${cy})`}
                role="button"
                tabIndex={0}
                data-stimulus-id={overlay.stimulusId}
                data-uploaded-marker="true"
                data-link-key={`uploaded:${overlay.stimulusId}`}
                data-roi={roi}
                data-linked-highlighted={isLinked ? 'true' : undefined}
                data-marker-color={UPLOADED_STIMULUS_MARKER_COLOR}
                data-assignment-method={overlay.assignmentMethod}
                aria-label={`Uploaded stimulus ${overlay.displayName}, ${overlayAssignmentDescription(overlay)}${
                  outsideDomain ? ', shown at the fixed reference boundary' : ''
                }`}
                aria-describedby={
                  dotTooltip?.kind === 'uploaded' && dotTooltip.overlay.stimulusId === overlay.stimulusId
                    ? tooltipId
                    : undefined
                }
                style={{ '--stimulus-color': overlay.color } as CSSProperties}
                onMouseEnter={(event) => {
                  setDotTooltip({ kind: 'uploaded', trigger: 'hover', overlay, ...tooltipAnchor(event.currentTarget) });
                  onLinkedItemChange?.({ kind: 'uploaded', id: overlay.stimulusId, label: overlay.displayName });
                }}
                onMouseLeave={clearUploadHover}
                onFocus={(event) => {
                  setDotTooltip({ kind: 'uploaded', trigger: 'focus', overlay, ...tooltipAnchor(event.currentTarget) });
                  onLinkedItemChange?.({ kind: 'uploaded', id: overlay.stimulusId, label: overlay.displayName });
                }}
                onBlur={() => {
                  setDotTooltip(null);
                  onLinkedItemChange?.(null);
                }}
                onClick={() => handleOverlayClick(overlay)}
                onKeyDown={(event) => {
                  if (event.key === 'Escape') {
                    setDotTooltip(null);
                    onLinkedItemChange?.(null);
                    return;
                  }
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleOverlayClick(overlay);
                  }
                }}
              >
                <circle className="session-cluster-marker-hit" r="12" />
                <circle className="session-cluster-marker-glow" r="7" fill={overlay.color} />
                <circle className="session-cluster-marker-focus" r="8.5" />
                <circle className="session-cluster-marker-core" r="4.6" fill={overlay.color} />
              </g>
            );
          })}
        </svg>

        {dotTooltip && createPortal(
          <div
            ref={tooltipRef}
            id={tooltipId}
            role="tooltip"
            className={`scatter-dot-tooltip ${dotTooltip.kind === 'uploaded' ? 'scatter-dot-tooltip--uploaded' : ''}`}
            style={{
              left: `clamp(${dotTooltip.kind === 'uploaded' ? 74 : 68}px, ${dotTooltip.anchorX}px, calc(100vw - ${
                dotTooltip.kind === 'uploaded' ? 74 : 68
              }px))`,
              top: `${dotTooltip.anchorY}px`,
            }}
          >
            <img
              src={dotTooltip.kind === 'reference' ? dotTooltip.point.imageUrl : dotTooltip.overlay.imageUrl}
              alt=""
              className="scatter-dot-tooltip-img"
            />
            <p className="scatter-dot-tooltip-label">
              {dotTooltip.kind === 'reference' ? dotTooltip.point.imageName : dotTooltip.overlay.displayName}
            </p>
            {dotTooltip.kind === 'reference' ? (
              <p className="scatter-dot-tooltip-sub">Cluster {dotTooltip.point.clusterLabel}</p>
            ) : (
              <p className="scatter-dot-tooltip-sub scatter-dot-tooltip-sub--uploaded">
                <span>Uploaded · Cluster {dotTooltip.overlay.clusterId}</span>
                {dotTooltip.overlay.assignmentMethod === 'joint hierarchical clustering' && <span>Joint hierarchical clustering</span>}
                <span>{overlayDistanceLabel(dotTooltip.overlay)} {dotTooltip.overlay.distance.toFixed(3)}</span>
                {dotTooltip.overlay.medoidImageName && <span>Medoid: {dotTooltip.overlay.medoidImageName}</span>}
              </p>
            )}
          </div>,
          document.body,
        )}
      </div>
      {hasOutsideOverlay && (
        <p className="cluster-boundary-note" role="note">
          Uploads outside the reference range are pinned to the plot boundary.
        </p>
      )}
    </figure>
  );
}

function Metrics({ summary }: { summary: ClusterSummary | null }) {
  if (!summary) return null;
  return (
    <dl className="summary-card-grid cluster-summary-grid">
      <div><dt>Silhouette</dt><dd>{fmtN(summary.silhouette)}</dd></div>
      {summary.nVoxels != null && <div><dt>Voxels</dt><dd>{fmtI(summary.nVoxels)}</dd></div>}
      {summary.nFeatures != null && <div><dt>Features</dt><dd>{fmtI(summary.nFeatures)}</dd></div>}
      {summary.pcaDimUsed != null && <div><dt>PCA dimensions</dt><dd>{fmtI(summary.pcaDimUsed)}</dd></div>}
    </dl>
  );
}

const CATEGORIES: ImageCategory[] = ['Faces', 'Places', 'Body Part'];

const CATEGORY_LABELS: Record<ImageCategory, string> = {
  'Faces': 'Faces',
  'Places': 'Places',
  'Body Part': 'Body parts',
};

function ClusterFilters({
  searchQuery,
  onSearchQueryChange,
  categoryHighlight,
  onCategoryHighlightChange,
  matchCount,
  label = 'scatter plots',
}: {
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  categoryHighlight: ImageCategory | null;
  onCategoryHighlightChange: (value: ImageCategory | null) => void;
  matchCount?: number;
  label?: string;
}) {
  const needle = searchQuery.trim();

  return (
    <div className="cluster-filter-controls">
      <div className="scatter-search-row">
        <input
          type="search"
          className="scatter-search-input"
          placeholder="Search image filename..."
          value={searchQuery}
          onChange={(event) => onSearchQueryChange(event.target.value)}
          aria-label={`Search image filenames in ${label}`}
        />
        {needle && matchCount !== undefined && (
          <span className="scatter-search-count">
            {matchCount} match{matchCount !== 1 ? 'es' : ''}
          </span>
        )}
      </div>
      <div className="scatter-category-row" role="group" aria-label={`Highlight ${label} by image category`}>
        <span className="scatter-category-label">Highlight:</span>
        {CATEGORIES.map((category) => (
          <button
            key={category}
            type="button"
            className={`scatter-category-btn scatter-category-btn--${category.toLowerCase().replace(' ', '-')} ${
              categoryHighlight === category ? 'active' : ''
            }`}
            onClick={() => onCategoryHighlightChange(categoryHighlight === category ? null : category)}
            aria-pressed={categoryHighlight === category}
          >
            {CATEGORY_LABELS[category]}
          </button>
        ))}
      </div>
    </div>
  );
}

function ClusterSection({
  title,
  description,
  view,
  emptyMessage,
  onOpenImage,
  imageCategories,
  overlays = [],
  compact,
  variant = 'full',
  footer,
  sharedSearchQuery,
  sharedCategoryHighlight,
  hideFilters = false,
  linkedItem,
  onLinkedItemChange,
  roi,
  session,
}: {
  title: string;
  description?: string;
  view: ClusterView | null;
  emptyMessage: string;
  onOpenImage: (i: EvidenceImage) => void;
  imageCategories: Map<string, ImageCategory>;
  overlays?: SessionClusterOverlay[];
  compact?: boolean;
  variant?: ClusterSectionVariant;
  footer?: ReactNode;
  sharedSearchQuery?: string;
  sharedCategoryHighlight?: ImageCategory | null;
  hideFilters?: boolean;
  linkedItem?: LinkedPlotItem | null;
  onLinkedItemChange?: (item: LinkedPlotItem | null) => void;
  roi?: CanonicalRoi;
  session?: CortexComparisonSession | null;
}) {
  const [localSearchQuery, setLocalSearchQuery] = useState('');
  const [localCategoryHighlight, setLocalCategoryHighlight] = useState<ImageCategory | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const searchQuery = sharedSearchQuery ?? localSearchQuery;
  const categoryHighlight = sharedCategoryHighlight === undefined ? localCategoryHighlight : sharedCategoryHighlight;
  const isBoard = variant === 'board';

  if (!view) {
    return compact ? (
      <>
        <p className="cluster-empty-note">{emptyMessage}</p>
        {footer}
      </>
    ) : (
      <section className={`cluster-section ${isBoard ? 'cluster-section--board' : ''}`}>
        {title && <h3>{title}</h3>}
        {description && <p className="cluster-section-description">{description}</p>}
        <p className="cluster-empty-note">{emptyMessage}</p>
        {footer}
      </section>
    );
  }

  const needle = searchQuery.trim().toLowerCase();
  const matchCount = needle ? view.points.filter((p) => p.imageName.toLowerCase().includes(needle)).length : 0;

  const inner = (
    <>
      {!compact && title && (
        <div className="cluster-section-heading">
          <div>
            <h3>{title}</h3>
            {description && <p>{description}</p>}
          </div>
        </div>
      )}
      <p className="cluster-section-meta">
        {view.groups.length} {view.groups.length === 1 ? 'cluster' : 'clusters'} · {view.summary?.nImages ?? view.points.length} images
        {view.jointHierarchy && ` (${view.jointHierarchy.referenceCount} reference + ${view.jointHierarchy.uploadedCount} uploaded)`}
      </p>
      <details className="cluster-details" onToggle={(event) => {
        const open = event.currentTarget.open;
        setDetailsOpen(open);
        if (!open) onLinkedItemChange?.(null);
      }}>
        <summary>Cluster details</summary>
        <div className="cluster-details-content">
          <p>{projectionLabel(view)} projection. Axes are fixed to the reference images.</p>
          {view.jointHierarchy && (
            <p>Joint clustering uses full response or feature vectors, not PCA coordinates. Reference normalization stays fixed; clusters and the tree are rebuilt with uploads.</p>
          )}
          <Metrics summary={view.summary} />
          {view.summary?.clusteringMethod && <p>Method: {view.summary.clusteringMethod}.</p>}
          {view.summary?.distanceMetric && <p>Distance: {view.summary.distanceMetric}.</p>}
          {view.summary?.normalization && <p>Normalization: {view.summary.normalization}.</p>}
          {view.summary?.distanceToClusterRepresentative && (
            <p>Image distances use {representativeDistanceLabel(view).toLowerCase()}.</p>
          )}
          {view.jointHierarchy && session ? detailsOpen && (
            <JointDendrogram
              view={view}
              session={session}
              onOpenImage={onOpenImage}
              onLinkedItemChange={onLinkedItemChange}
              linkedItem={linkedItem}
            />
          ) : view.summary?.dendrogramPng && (
            <figure className="cluster-dendrogram">
              <img src={view.summary.dendrogramPng} alt={`${title || 'Selected clustering'} dendrogram`} loading="lazy" />
              <figcaption>
                {view.summary.distanceMetric ?? 'correlation'} distance
                {view.summary.linkageMethod ? ` · ${view.summary.linkageMethod} linkage` : ''}
              </figcaption>
            </figure>
          )}
        </div>
      </details>

      {!hideFilters && (
        <ClusterFilters
          searchQuery={searchQuery}
          onSearchQueryChange={setLocalSearchQuery}
          categoryHighlight={categoryHighlight}
          onCategoryHighlightChange={setLocalCategoryHighlight}
          matchCount={needle ? matchCount : undefined}
          label="scatter plot"
        />
      )}

      <ClusterScatter
        view={view}
        onOpenImage={onOpenImage}
        searchQuery={searchQuery}
        categoryHighlight={categoryHighlight}
        imageCategories={imageCategories}
        overlays={overlays}
        variant={variant}
        linkedItem={linkedItem}
        onLinkedItemChange={onLinkedItemChange}
        roi={roi}
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
      {footer}
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
        <h3>Measured fMRI clusters</h3>
        <p className="cluster-section-description">Images grouped by measured participant responses.</p>
        {sessionActive && (
          <p className="ground-truth-session-note" data-no-upload-overlay="true">
            Uploads are predictions only; this view uses measured fMRI.
          </p>
        )}
        <p className="cluster-empty-note">Measured fMRI clusters are unavailable.</p>
      </section>
    );
  }

  const patient = patients.find((p) => p.id === activePatient) ?? patients[0];
  const selectedRoi = activeRoi ?? patient.rois[0]?.roi ?? null;
  const roiView = patient.rois.find((r) => r.roi === selectedRoi);

  return (
    <section className={`cluster-section ${variant === 'board' ? 'cluster-section--board' : ''}`}>
      <h3>Measured fMRI clusters</h3>
      <p className="cluster-section-description">Images grouped by measured participant responses.</p>
      {sessionActive && (
        <p className="ground-truth-session-note" data-no-upload-overlay="true">
          Uploads are predictions only; this view uses measured fMRI.
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
  const results = session.roiResults[roi].stimuli;
  const jointHierarchy = session.roiResults[roi].jointHierarchy;
  if (method === 'hierarchical_correlation' && !jointHierarchy) return null;
  const jointUploads = new Map(jointHierarchy?.members
    .filter((member) => member.kind === 'uploaded').map((member) => [member.id, member]));
  const stimuli = new Map(session.stimuli.map((stimulus) => [stimulus.id, stimulus]));
  const focusedResult = results.find((result) => result.stimulusId === focusedStimulusId) ?? results[0];
  const focusedStimulus = session.stimuli.find((stimulus) => stimulus.id === focusedResult?.stimulusId);
  const focusedCluster = focusedResult
    ? method === 'kmeans_euclidean'
      ? focusedResult.voxelProjection.clusterId
      : jointUploads.get(focusedResult.stimulusId)?.clusterId ?? null
    : null;

  return (
    <details className="session-cluster-assignments">
      <summary>
        {focusedStimulus && <span className="stimulus-dot" style={{ backgroundColor: UPLOADED_STIMULUS_MARKER_COLOR }} />}
        <span>
          {results.length === 1 && focusedStimulus
            ? `Uploaded: ${focusedStimulus.displayName}`
            : `${results.length} uploaded ${method === 'hierarchical_correlation' ? 'images' : 'placements'}`}
        </span>
        {focusedCluster !== null && <strong>Cluster {focusedCluster}</strong>}
      </summary>
      <div className="session-assignment-details">
        {method === 'hierarchical_correlation' && (
          <p>
            Reference images and uploads are clustered together using full predicted-response vectors.
            Medoids summarize the resulting clusters; they do not assign membership.
          </p>
        )}
        <div className="session-assignment-list">
          {results.map((result) => {
            const stimulus = stimuli.get(result.stimulusId);
            if (!stimulus) return null;
            const member = jointUploads.get(result.stimulusId);
            return (
              <article key={stimulus.id} className={stimulus.id === focusedStimulusId ? 'focused' : ''}>
                <div>
                  <span className="stimulus-dot" style={{ backgroundColor: UPLOADED_STIMULUS_MARKER_COLOR }} />
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
                ) : member && jointHierarchy ? (
                  <>
                    <span>Joint hierarchical clustering</span>
                    <span>Cluster {member.clusterId} · medoid correlation distance {member.distanceToMedoid.toFixed(3)}</span>
                    <small>Medoid: {jointClusterMedoidName(jointHierarchy, member.clusterId, session)}</small>
                  </>
                ) : null}
              </article>
            );
          })}
        </div>
      </div>
    </details>
  );
}

export function VoxelRoiComparison({
  voxelRoiStates,
  clusteringMethod,
  imageCategories,
  onOpenImage,
  session,
  selectedRoi,
  focusedStimulusId,
}: {
  voxelRoiStates: Record<CanonicalRoi, VoxelRoiClusterState>;
  clusteringMethod: ClusteringMethod;
  imageCategories: Map<string, ImageCategory>;
  onOpenImage: (image: EvidenceImage) => void;
  session: CortexComparisonSession | null;
  selectedRoi: CanonicalRoi | null;
  focusedStimulusId: string | null;
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryHighlight, setCategoryHighlight] = useState<ImageCategory | null>(null);
  const [linkedItem, setLinkedItem] = useState<LinkedPlotItem | null>(null);
  const displayStates = useMemo(
    () => jointRoiStates(voxelRoiStates, clusteringMethod, session),
    [voxelRoiStates, clusteringMethod, session],
  );
  const overlaysByRoi = useMemo(() => Object.fromEntries(ROI_COMPARISON_ORDER.map((roi) => {
    const view = displayStates[roi].view;
    return [roi, session && (clusteringMethod !== 'hierarchical_correlation' || view?.jointHierarchy)
      ? buildVoxelSessionOverlays(session, roi, clusteringMethod, view?.uploadAxisAlignment)
      : []];
  })) as Record<CanonicalRoi, SessionClusterOverlay[]>, [displayStates, clusteringMethod, session]);
  const needle = searchQuery.trim().toLowerCase();
  const matchedNames = new Set<string>();

  if (needle) {
    for (const roi of ROI_COMPARISON_ORDER) {
      for (const point of displayStates[roi].view?.points ?? []) {
        if (point.imageName.toLowerCase().includes(needle)) matchedNames.add(point.imageName);
      }
    }
  }

  return (
    <section className="roi-cluster-comparison" aria-labelledby="roi-cluster-comparison-title">
      <div className="roi-cluster-comparison-heading">
        <div>
          <h3 id="roi-cluster-comparison-title">Predicted-response clusters across ROIs</h3>
          <p>Hover a point to trace the same image across FFA, EBA, and PPA.</p>
        </div>
        <span className={`roi-cluster-linked-label ${linkedItem ? 'active' : ''}`}>
          {linkedItem
            ? `${linkedItem.kind === 'uploaded' ? 'Uploaded' : 'Linked'}: ${linkedItem.label}`
            : 'Linked hover ready'}
        </span>
      </div>

      <ClusterFilters
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        categoryHighlight={categoryHighlight}
        onCategoryHighlightChange={setCategoryHighlight}
        matchCount={needle ? matchedNames.size : undefined}
        label="FFA, EBA, and PPA scatter plots"
      />

      <div className="roi-cluster-grid">
        {ROI_COMPARISON_ORDER.map((roi) => {
          const state = displayStates[roi];
          const overlays = overlaysByRoi[roi];
          const note = session && clusteringMethod === 'hierarchical_correlation'
            ? jointClusteringNote(state.view, Boolean(session.roiResults[roi].jointHierarchy))
            : null;
          const footer = session && (clusteringMethod !== 'hierarchical_correlation' || state.view?.jointHierarchy) ? (
            <SessionVoxelAssignments
              session={session}
              roi={roi}
              method={clusteringMethod}
              focusedStimulusId={focusedStimulusId}
            />
          ) : null;

          return (
            <div
              key={roi}
              className={`roi-cluster-card ${selectedRoi === roi ? 'roi-cluster-card--selected' : ''}`}
              data-roi-cluster-chart={roi}
              aria-label={`${roi.toUpperCase()} predicted-response clusters`}
            >
              {state.status === 'loading' ? (
                <section className="cluster-section cluster-section--board">
                  <div className="cluster-section-heading">
                    <div><h3>{roi.toUpperCase()}</h3><p>Images grouped by predicted response patterns.</p></div>
                  </div>
                  <p className="cluster-empty-note">Loading…</p>
                </section>
              ) : (
                <ClusterSection
                  title={roi.toUpperCase()}
                  description="Images grouped by predicted response patterns."
                  view={state.status === 'error' ? null : state.view}
                  emptyMessage={state.status === 'error' ? 'Response clusters could not be loaded.' : 'Response clusters are unavailable for this ROI.'}
                  imageCategories={imageCategories}
                  onOpenImage={onOpenImage}
                  overlays={overlays}
                  variant="board"
                  footer={<>
                    {note && <p className="session-cluster-note">{note}</p>}
                    {footer}
                  </>}
                  sharedSearchQuery={searchQuery}
                  sharedCategoryHighlight={categoryHighlight}
                  hideFilters
                  linkedItem={linkedItem}
                  onLinkedItemChange={setLinkedItem}
                  roi={roi}
                  session={session}
                />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function ClusteringPanel({
  voxelRoiStates,
  clusteringMethod,
  clusteringMethodOptions,
  onClusteringMethodChange,
  visualView,
  dreamsimView,
  imageCategories,
  groundTruthPatients,
  onOpenImage,
  session: providedSession,
  selectedModel,
  selectedRoi,
  selectivityAtlasAvailable,
  showSelectivityLauncher = true,
  focusedStimulusId,
}: ClusteringPanelProps) {
  const [isRoiComparisonOpen, setIsRoiComparisonOpen] = useState(false);
  const roiComparisonTriggerRef = useRef<HTMLButtonElement | null>(null);
  const roi = isCanonicalRoi(selectedRoi) ? selectedRoi : null;
  const session = providedSession?.comparison.model === selectedModel ? providedSession : null;
  const displayStates = useMemo(
    () => jointRoiStates(voxelRoiStates, clusteringMethod, session),
    [voxelRoiStates, clusteringMethod, session],
  );
  const displayVisualView = useMemo(() => {
    const hierarchy = session?.visualResults?.jointHierarchy;
    return clusteringMethod === 'hierarchical_correlation' && visualView && hierarchy && session
      ? buildJointClusterView(visualView, hierarchy, session) ?? visualView
      : visualView;
  }, [visualView, session, clusteringMethod]);
  const voxelState = roi ? displayStates[roi] : { view: null, status: 'idle' as const };
  const voxelOverlays = session && roi && (clusteringMethod !== 'hierarchical_correlation' || voxelState.view?.jointHierarchy)
    ? buildVoxelSessionOverlays(session, roi, clusteringMethod, voxelState.view?.uploadAxisAlignment)
    : [];
  const visualOverlays = session && (clusteringMethod !== 'hierarchical_correlation' || displayVisualView?.jointHierarchy)
    ? buildOptionalSessionOverlays(session, session.visualResults, clusteringMethod, displayVisualView?.uploadAxisAlignment)
    : [];
  const voxelNote = session && roi && clusteringMethod === 'hierarchical_correlation'
    ? jointClusteringNote(voxelState.view, Boolean(session.roiResults[roi].jointHierarchy))
    : null;

  function closeRoiComparison() {
    setIsRoiComparisonOpen(false);
    window.requestAnimationFrame(() => roiComparisonTriggerRef.current?.focus());
  }

  return (
    <div className="clustering-panel">
      <div className="cluster-toolbar">
        <ClusteringMethodControl
          value={clusteringMethod}
          options={clusteringMethodOptions}
          onChange={onClusteringMethodChange}
        />
        <button
          ref={roiComparisonTriggerRef}
          type="button"
          className="roi-compare-trigger"
          aria-haspopup="dialog"
          aria-expanded={isRoiComparisonOpen}
          onClick={() => setIsRoiComparisonOpen(true)}
        >
          Compare FFA · EBA · PPA
        </button>
        {roi && showSelectivityLauncher && (
          <VoxelSelectivityAtlasDialog
            model={selectedModel}
            available={selectivityAtlasAvailable}
          />
        )}
      </div>
      <p className="cluster-panel-intro">Each dot is an image; colors mark clusters. Hover to preview; click to enlarge.</p>
      {session && <p className="cluster-panel-upload-key">
        {clusteringMethod === 'hierarchical_correlation'
          ? 'Reference images and uploads are clustered together when joint data are available. Glowing dots are uploads; display axes stay fixed.'
          : 'Glowing dots are uploads; reference axes stay fixed.'}
      </p>}
      <div className="cluster-board">
        {voxelState.status === 'loading' ? (
          <section className="cluster-section cluster-section--board">
            <h3>Predicted-response clusters</h3>
            <p className="cluster-section-description">Images grouped by predicted voxel-response patterns.</p>
            <p className="cluster-empty-note">Loading…</p>
          </section>
        ) : (
          <ClusterSection
            title="Predicted-response clusters"
            description="Images grouped by predicted voxel-response patterns."
            view={voxelState.status === 'error' ? null : voxelState.view}
            emptyMessage={
              voxelState.status === 'error'
                ? 'Response clusters could not be loaded.'
                : 'Response clusters are unavailable for this model and ROI.'
            }
            imageCategories={imageCategories}
            onOpenImage={onOpenImage}
            overlays={voxelOverlays}
            variant="board"
            footer={session && roi ? (
              <>
                {voxelNote && <p className="session-cluster-note">{voxelNote}</p>}
                {(clusteringMethod !== 'hierarchical_correlation' || voxelState.view?.jointHierarchy) && <SessionVoxelAssignments
                  session={session}
                  roi={roi}
                  method={clusteringMethod}
                  focusedStimulusId={focusedStimulusId}
                />}
              </>
            ) : null}
            session={session}
          />
        )}
        <ClusterSection
          title="Visual-feature clusters"
          description="Images grouped by color, texture, and other visual features."
          view={displayVisualView}
          emptyMessage="Visual-feature clusters are unavailable."
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          overlays={visualOverlays}
          variant="board"
          session={session}
          footer={session && visualOverlays.length === 0 ? (
            <p className="session-cluster-note">
              {clusteringMethod === 'hierarchical_correlation'
                ? jointClusteringNote(displayVisualView, Boolean(session.visualResults?.jointHierarchy))
                  ?? 'Upload placement is unavailable for this view.'
                : 'Upload placement is unavailable for this view.'}
            </p>
          ) : null}
        />
        <ClusterSection
          title="DreamSim similarity clusters"
          description="Images grouped by DreamSim perceptual embeddings."
          view={dreamsimView}
          emptyMessage="DreamSim clusters are unavailable."
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          variant="board"
          footer={session ? (
            <details className="session-cluster-note" data-dreamsim-overlay="disabled">
              <summary>Uploaded placement unavailable</summary>
              <p>
                The archived reference bundle has 1,792 features, while the current Cortex embedding path produces 768.
                No incompatible DreamSim projection is shown.
              </p>
            </details>
          ) : null}
        />
        <GroundTruthSection
          patients={groundTruthPatients}
          imageCategories={imageCategories}
          onOpenImage={onOpenImage}
          sessionActive={Boolean(session)}
          variant="board"
        />
      </div>
      {isRoiComparisonOpen ? createPortal(
        <div
          className="roi-comparison-modal-backdrop"
          role="presentation"
          onClick={closeRoiComparison}
          onKeyDown={(event) => {
            if (event.key === 'Escape') closeRoiComparison();
          }}
        >
          <section
            className="roi-comparison-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="roi-comparison-modal-title"
            onClick={(event) => event.stopPropagation()}
            onKeyDown={keepFocusInsideDialog}
          >
            <header className="roi-comparison-modal-header">
              <div>
                <p className="eyebrow">Linked image comparison</p>
                <h2 id="roi-comparison-modal-title">FFA · EBA · PPA</h2>
                <p>Predicted-response clusters across ROIs, using the same model and method.</p>
              </div>
              <button type="button" autoFocus onClick={closeRoiComparison}>Close</button>
            </header>
            <VoxelRoiComparison
              voxelRoiStates={displayStates}
              clusteringMethod={clusteringMethod}
              imageCategories={imageCategories}
              onOpenImage={onOpenImage}
              session={session}
              selectedRoi={roi}
              focusedStimulusId={focusedStimulusId}
            />
          </section>
        </div>,
        document.body,
      ) : null}
    </div>
  );
}
