import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { loadSelectivityAtlasAssets } from '../services/selectivityAtlasAssets';
import type {
  SelectivityAtlasAssets,
  SelectivityAtlasCluster,
  SelectivityAtlasHoverDatum,
  SelectivityAtlasRoi,
  SelectivityAtlasStimulus,
} from '../types/selectivityAtlas';
import {
  SELECTIVITY_CLUSTER_COLORS,
  SELECTIVITY_ROI_LABELS,
  selectivityImageUrl,
} from '../utils/selectivityAtlas';

const VoxelSelectivityAtlasViewer = lazy(() => import('./VoxelSelectivityAtlasViewer'));
const ROI_ORDER: SelectivityAtlasRoi[] = ['ffa', 'eba', 'ppa'];

type AssetState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; assets: SelectivityAtlasAssets };

type VoxelSelectivityAtlasDialogProps = {
  model: string;
  available: boolean;
};

function keepFocusInsideDialog(event: KeyboardEvent<HTMLElement>) {
  if (event.key !== 'Tab') return;
  const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
    'button:not([disabled]), input:not([disabled]), a[href], summary, [tabindex]:not([tabindex="-1"])',
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

function StimulusStrip({ title, stimuli }: { title: string; stimuli: SelectivityAtlasStimulus[] }) {
  return (
    <section className="selectivity-stimulus-strip">
      <h4>{title}</h4>
      <div className="selectivity-stimulus-grid">
        {stimuli.map((stimulus) => (
          <figure key={stimulus.imageId} title={stimulus.imageId}>
            <img src={selectivityImageUrl(stimulus.imageId)} alt={stimulus.imageId} />
            <figcaption>
              <strong>{stimulus.meanZ >= 0 ? '+' : ''}{stimulus.meanZ.toFixed(2)}</strong>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

function ClusterProfile({ cluster }: { cluster: SelectivityAtlasCluster | null }) {
  if (!cluster) {
    return (
      <section className="selectivity-cluster-profile selectivity-cluster-profile--empty">
        <strong>Select a cluster</strong>
        <p>Click a dot or cluster label to see its highest- and lowest-response reference images.</p>
      </section>
    );
  }
  return (
    <section className="selectivity-cluster-profile">
      <header>
        <span
          className="selectivity-cluster-swatch"
          style={{ backgroundColor: SELECTIVITY_CLUSTER_COLORS[cluster.label] }}
        />
        <div>
          <h3>Cluster C{cluster.label}</h3>
          <p>
            {cluster.size} voxels · FFA {cluster.roiCounts.ffa} · EBA {cluster.roiCounts.eba} · PPA {cluster.roiCounts.ppa}
          </p>
        </div>
        <span className="selectivity-cluster-response-label">Mean predicted response (z)</span>
      </header>
      <div className="selectivity-profile-strips">
        <StimulusStrip title="Highest response" stimuli={cluster.preferredImages} />
        <StimulusStrip title="Lowest response" stimuli={cluster.suppressedImages} />
      </div>
    </section>
  );
}

function VoxelSelectivityAtlasModal({
  model,
  onClose,
}: Omit<VoxelSelectivityAtlasDialogProps, 'available'> & { onClose: () => void }) {
  const [assetState, setAssetState] = useState<AssetState>({ status: 'loading' });
  const [enabledRois, setEnabledRois] = useState<Record<SelectivityAtlasRoi, boolean>>({
    ffa: true,
    eba: true,
    ppa: true,
  });
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [hovered, setHovered] = useState<SelectivityAtlasHoverDatum | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const enabledRoiSet = useMemo(
    () => new Set(ROI_ORDER.filter((roi) => enabledRois[roi])),
    [enabledRois],
  );

  useEffect(() => {
    let ignore = false;
    const abortController = new AbortController();
    loadSelectivityAtlasAssets(model, fetch, abortController.signal)
      .then((assets) => {
        if (!ignore) setAssetState({ status: 'ready', assets });
      })
      .catch((error: unknown) => {
        if (ignore) return;
        setAssetState({
          status: 'error',
          message: error instanceof Error ? error.message : 'Voxel selectivity could not be loaded.',
        });
      });
    return () => {
      ignore = true;
      abortController.abort();
    };
  }, [model]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previousOverflow; };
  }, []);

  const selectedProfile = assetState.status === 'ready' && selectedCluster !== null
    ? assetState.assets.summary.clusters.find((cluster) => cluster.label === selectedCluster) ?? null
    : null;
  const visibleCount = assetState.status === 'ready'
    ? ROI_ORDER.reduce(
        (sum, roi) => sum + (enabledRois[roi] ? assetState.assets.manifest.rois[roi].count : 0),
        0,
      )
    : 0;

  return createPortal(
    <div
      className="selectivity-atlas-modal-backdrop"
      role="presentation"
      onClick={onClose}
      onKeyDown={(event) => {
        if (event.key === 'Escape') onClose();
      }}
    >
      <section
        className="selectivity-atlas-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="selectivity-atlas-modal-title"
        aria-describedby="selectivity-atlas-modal-description"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={keepFocusInsideDialog}
      >
        <header className="selectivity-atlas-modal-header">
          <div>
            <p className="eyebrow">NSD Subject 1 · model predictions</p>
            <h2 id="selectivity-atlas-modal-title">Voxel Selectivity Explorer</h2>
            <p id="selectivity-atlas-modal-description">
              {model} · 185 Murty reference images · uploads not included
            </p>
          </div>
          <button type="button" autoFocus onClick={onClose}>Close</button>
        </header>

        {assetState.status === 'loading' && (
          <div className="selectivity-atlas-load-state" role="status">
            <span className="selectivity-atlas-spinner" aria-hidden="true" />
            <strong>Loading voxel selectivity…</strong>
          </div>
        )}

        {assetState.status === 'error' && (
          <div className="selectivity-atlas-load-state error" role="alert">
            <strong>Voxel selectivity unavailable</strong>
            <p>{assetState.message}</p>
          </div>
        )}

        {assetState.status === 'ready' && (
          <>
            <div className="selectivity-atlas-layout">
              <aside className="selectivity-atlas-controls" aria-label="Voxel selectivity controls">
                <fieldset>
                  <legend>Brain regions</legend>
                  {ROI_ORDER.map((roi) => (
                    <label key={roi}>
                      <input
                        type="checkbox"
                        checked={enabledRois[roi]}
                        onChange={(event) => setEnabledRois((current) => ({
                          ...current,
                          [roi]: event.target.checked,
                        }))}
                      />
                      <span><strong>{roi.toUpperCase()}</strong> — {SELECTIVITY_ROI_LABELS[roi]}</span>
                      <small>{assetState.assets.manifest.rois[roi].count}</small>
                    </label>
                  ))}
                  <small>Visibility only; clusters use all three regions.</small>
                </fieldset>

                <fieldset>
                  <legend>{assetState.assets.summary.clusterCount} exploratory clusters</legend>
                  <div className="selectivity-cluster-legend">
                    {assetState.assets.summary.clusters.map((cluster) => (
                      <button
                        key={cluster.label}
                        type="button"
                        className={selectedCluster === cluster.label ? 'active' : ''}
                        aria-pressed={selectedCluster === cluster.label}
                        aria-label={`Cluster C${cluster.label}, ${cluster.size} voxels`}
                        style={{ '--cluster-color': SELECTIVITY_CLUSTER_COLORS[cluster.label] } as CSSProperties}
                        onClick={() => setSelectedCluster((current) => (
                          current === cluster.label ? null : cluster.label
                        ))}
                      >
                        <span aria-hidden="true" />
                        C{cluster.label}
                        <small>{cluster.size}</small>
                      </button>
                    ))}
                  </div>
                </fieldset>

                <details className="selectivity-atlas-method">
                  <summary>Method &amp; quality</summary>
                  <p>
                    Each voxel’s predictions are z-scored across images, reduced to{' '}
                    {assetState.assets.summary.method.pcaComponents} PCA dimensions, and L2-normalized.
                    K-means groups these profiles into {assetState.assets.summary.clusterCount} clusters;
                    UMAP provides the 3D display only.
                  </p>
                  <dl className="selectivity-atlas-diagnostics">
                    <div><dt>Silhouette</dt><dd>{assetState.assets.summary.silhouette.toFixed(3)}</dd></div>
                    <div>
                      <dt>Split-half ARI · mean / min</dt>
                      <dd>
                        {assetState.assets.summary.splitHalfStability.meanAdjustedRandIndex.toFixed(3)}
                        {' / '}{assetState.assets.summary.splitHalfStability.minAdjustedRandIndex.toFixed(3)}
                      </dd>
                    </div>
                    {selectedProfile?.meanSilhouette != null && (
                      <div><dt>C{selectedProfile.label} silhouette</dt><dd>{selectedProfile.meanSilhouette.toFixed(3)}</dd></div>
                    )}
                  </dl>
                  <p>
                    Silhouette measures cluster separation. ARI measures agreement across{' '}
                    {assetState.assets.summary.splitHalfStability.iterations} stimulus split-half checks.
                    These describe this fit; the fixed cluster count is exploratory, not a validated optimum.
                  </p>
                  <p>
                    Image scores average each voxel’s standardized prediction within a cluster.
                    They emphasize relative preference, not absolute activity or inhibition.
                    These are model predictions, not measured fMRI responses.
                  </p>
                  <p>
                    The reference set contains 110 places, 50 body parts, and 25 faces.
                    This imbalance and differences between ROIs can shape the clusters.
                    Cluster colors and numbers are specific to each model; the groups are not anatomical parcels.
                  </p>
                </details>
              </aside>

              <div className="selectivity-atlas-stage-column">
                <div className="selectivity-atlas-stage-heading">
                  <p>Dots: voxels · colors: response-profile clusters · positions: similarity, not anatomy.</p>
                  <span>{visibleCount} / {assetState.assets.summary.voxelCount} voxels visible</span>
                </div>
                <div className="selectivity-atlas-stage">
                  <Suspense fallback={<div className="selectivity-atlas-canvas-state">Preparing the 3D renderer…</div>}>
                    <VoxelSelectivityAtlasViewer
                      points={assetState.assets.points}
                      enabledRois={enabledRoiSet}
                      selectedCluster={selectedCluster}
                      resetKey={resetKey}
                      onHover={setHovered}
                      onSelectCluster={(cluster) => setSelectedCluster((current) => current === cluster ? null : cluster)}
                    />
                  </Suspense>
                  {visibleCount === 0 && (
                    <div className="selectivity-atlas-canvas-state" role="status">
                      Select a brain region to show its voxels.
                    </div>
                  )}
                  {hovered && (
                    <div
                      className="selectivity-atlas-tooltip"
                      style={{ left: hovered.canvasX, top: hovered.canvasY } as CSSProperties}
                      role="status"
                    >
                      <strong>{hovered.roi.toUpperCase()} voxel {hovered.roiVoxelIndex}</strong>
                      <span>Cluster C{hovered.clusterLabel}</span>
                      <span>PCA centroid distance {hovered.distanceToCentroid.toFixed(3)}</span>
                    </div>
                  )}
                </div>
                <div className="selectivity-atlas-stage-footer">
                  <span>Drag to rotate · scroll to zoom · click a dot to select</span>
                  <button
                    type="button"
                    className="selectivity-atlas-reset"
                    onClick={() => setResetKey((key) => key + 1)}
                  >
                    Reset view
                  </button>
                </div>
              </div>
            </div>

            <ClusterProfile cluster={selectedProfile} />
          </>
        )}
      </section>
    </div>,
    document.body,
  );
}

export function VoxelSelectivityAtlasDialog({
  model,
  available,
}: VoxelSelectivityAtlasDialogProps) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  function close() {
    setIsOpen(false);
    window.requestAnimationFrame(() => triggerRef.current?.focus());
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="selectivity-atlas-trigger"
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        disabled={!available}
        title={available
          ? 'Explore voxels with similar predicted responses across FFA, EBA, and PPA'
          : 'Voxel selectivity is not available for this model'}
        onClick={() => setIsOpen(true)}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="7" cy="8" r="2" />
          <circle cx="15.5" cy="6" r="2" />
          <circle cx="17" cy="15" r="2" />
          <circle cx="8" cy="17" r="2" />
          <path d="M8.8 7.4 13.6 6.4M16 8l.7 5M15.3 15.5l-5.4 1M7.5 15l-.3-5" />
        </svg>
        Voxel Selectivity Explorer
      </button>
      {isOpen && (
        <VoxelSelectivityAtlasModal key={model} model={model} onClose={close} />
      )}
    </>
  );
}
