import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ClusteringPanel } from './ClusteringPanel';
import { ImageGrid } from './ImageGrid';
import { SessionEvidencePanel } from './SessionEvidencePanel';
import { SpatialTools } from './SpatialTools';
import type {
  AggregateHeatmapCell,
  ClusteringData,
  ClusteringMethod,
  ClusterView,
  EvidenceImage,
  ModelRoiColumn,
  SelectedHeatmapCell,
  WideCsvRow,
} from '../types/data';
import type { CanonicalRoi, CortexComparisonSession } from '../types/session';
import { copyText, downloadJson } from '../utils/browserActions';
import { CLUSTERING_METHODS, findVoxelClusterSummary, loadVoxelClusterView } from '../utils/clustering';
import { buildEvidenceView } from '../utils/evidence';
import { inferModelCategory } from '../utils/modelTags';
import { buildSessionEvidenceView } from '../utils/sessionEvidence';
import { buildJointClusterView } from '../utils/jointClustering';

type DetailsPanelProps = {
  heatmapCells: AggregateHeatmapCell[];
  imageCount: number;
  clustering: ClusteringData;
  modelRoiColumns: ModelRoiColumn[];
  onSelectCell: (cell: SelectedHeatmapCell) => void;
  rows: WideCsvRow[];
  selectedCell: SelectedHeatmapCell | null;
  session: CortexComparisonSession | null;
  focusedStimulusId: string | null;
  onFocusStimulus: (stimulusId: string) => void;
};

const topKOptions = [6, 9, 12];
const voxelRois: CanonicalRoi[] = ['ffa', 'eba', 'ppa'];
type DetailTab = 'evidence' | 'clustering' | 'spatial';
type VoxelLoadStatus = 'idle' | 'loading' | 'error';
type VoxelRoiStates = Record<CanonicalRoi, { view: ClusterView | null; status: VoxelLoadStatus }>;

function makeVoxelRoiStates(status: VoxelLoadStatus = 'idle'): VoxelRoiStates {
  return {
    ffa: { view: null, status },
    eba: { view: null, status },
    ppa: { view: null, status },
  };
}

function fmt(score: number | null) { return score === null ? '—' : score.toFixed(3); }
function fmtV(v: number)           { return v.toFixed(3); }

function ImageModal({ image, onClose }: { image: EvidenceImage; onClose: () => void }) {
  const [missing, setMissing] = useState(false);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const activeElement = document.activeElement;
    const underlyingDialog = Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"][aria-modal="true"]'))
      .find((dialog) => !dialog.contains(closeButtonRef.current));
    const previousFocus = activeElement === document.body
      ? underlyingDialog?.querySelector<HTMLElement>('button:not([disabled]), [tabindex="0"]') ?? activeElement
      : activeElement;
    closeButtonRef.current?.focus();
    return () => {
      if (previousFocus instanceof HTMLElement || previousFocus instanceof SVGElement) {
        if (previousFocus.isConnected) previousFocus.focus();
      }
    };
  }, []);

  return createPortal(
    <div className="image-modal-backdrop" role="presentation" onClick={onClose}
         onKeyDown={(event) => {
           if (event.key === 'Escape') {
             event.preventDefault();
             event.stopPropagation();
             onClose();
           } else if (event.key === 'Tab') {
             event.preventDefault();
             closeButtonRef.current?.focus();
           }
         }}>
      <div className="image-modal" role="dialog" aria-modal="true" aria-label={image.imageName}
           onClick={(e) => e.stopPropagation()}>
        <div className="image-modal-header">
          <div>
            <h3>{image.imageName}</h3>
            <p>{image.valueLabel ?? `Value: ${fmtV(image.value)}`}</p>
          </div>
          <button ref={closeButtonRef} type="button" onClick={onClose}>Close</button>
        </div>
        {missing
          ? <span className="modal-missing-image">Image unavailable</span>
          : <img src={image.imageUrl} alt={image.imageName} onError={() => setMissing(true)} />}
      </div>
    </div>,
    document.body,
  );
}

export function DetailsPanel({
  heatmapCells: _heatmapCells,
  imageCount: _imageCount,
  clustering,
  modelRoiColumns,
  onSelectCell: _onSelectCell,
  rows,
  selectedCell,
  session,
  focusedStimulusId,
  onFocusStimulus,
}: DetailsPanelProps) {
  const [topK, setTopK] = useState(6);
  const [activeTab, setActiveTab] = useState<DetailTab>('clustering');
  const [modalImage, setModalImage] = useState<EvidenceImage | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const [voxelRoiStates, setVoxelRoiStates] = useState<VoxelRoiStates>(() => makeVoxelRoiStates());
  const [clusteringMethod, setClusteringMethod] = useState<ClusteringMethod>('kmeans_euclidean');

  const evidenceView = useMemo(
    () => (selectedCell ? buildEvidenceView(selectedCell, rows, modelRoiColumns, topK) : null),
    [modelRoiColumns, rows, selectedCell, topK],
  );
  const sessionEvidenceView = useMemo(
    () => (selectedCell && session
      ? buildSessionEvidenceView(selectedCell, rows, modelRoiColumns, session, focusedStimulusId, topK)
      : null),
    [focusedStimulusId, modelRoiColumns, rows, selectedCell, session, topK],
  );

  const activeVoxelSummaries = clustering.voxelSummariesByMethod[clusteringMethod] ?? clustering.voxelSummaries;
  const activeVisualSummary = clustering.visualSummaryByMethod[clusteringMethod] ?? clustering.visualSummary;
  const activeVisualView = clustering.visualViewByMethod[clusteringMethod] ?? clustering.visualView;
  const activeDreamsimSummary = clustering.dreamsimSummaryByMethod[clusteringMethod] ?? clustering.dreamsimSummary;
  const activeDreamsimView = clustering.dreamsimViewByMethod[clusteringMethod] ?? clustering.dreamsimView;
  const activeGroundTruthPatients = clustering.groundTruthPatientsByMethod[clusteringMethod] ?? clustering.groundTruthPatients;

  const voxelSummary = useMemo(
    () => (selectedCell ? findVoxelClusterSummary(activeVoxelSummaries, selectedCell) : null),
    [activeVoxelSummaries, selectedCell],
  );
  const selectedVoxelRoi = selectedCell
    ? voxelRois.find((roi) => roi === selectedCell.roi) ?? null
    : null;

  useEffect(() => {
    if (selectedCell) {
      setActiveTab(sessionEvidenceView ? 'evidence' : 'clustering');
    }
  }, [selectedCell?.id, Boolean(sessionEvidenceView)]);

  useEffect(() => {
    let ignore = false;
    const selectedModel = selectedCell?.model;
    if (!selectedModel) {
      setVoxelRoiStates(makeVoxelRoiStates());
      return () => { ignore = true; };
    }

    setVoxelRoiStates(makeVoxelRoiStates('loading'));

    voxelRois.forEach((roi) => {
      loadVoxelClusterView(activeVoxelSummaries, { model: selectedModel, roi })
        .then((view) => {
          if (ignore) return;
          setVoxelRoiStates((current) => ({
            ...current,
            [roi]: { view, status: 'idle' },
          }));
        })
        .catch(() => {
          if (ignore) return;
          setVoxelRoiStates((current) => ({
            ...current,
            [roi]: { view: null, status: 'error' },
          }));
        });
    });

    return () => { ignore = true; };
  }, [activeVoxelSummaries, selectedCell?.model]);

  function payload() {
    if (!selectedCell) return null;
    const activeSession = session?.comparison.model === selectedCell.model ? session : null;
    const views = Object.fromEntries(voxelRois.map((roi) => {
      const referenceView = voxelRoiStates[roi].view;
      const hierarchy = activeSession?.roiResults[roi].jointHierarchy;
      const view = clusteringMethod === 'hierarchical_correlation' && referenceView && hierarchy && activeSession
        ? buildJointClusterView(referenceView, hierarchy, activeSession) ?? referenceView
        : referenceView;
      return [roi, view];
    })) as Record<CanonicalRoi, ClusterView | null>;
    const activeView = selectedVoxelRoi ? views[selectedVoxelRoi] : null;
    const visualHierarchy = activeSession?.visualResults?.jointHierarchy;
    const visualView = clusteringMethod === 'hierarchical_correlation' && activeVisualView && visualHierarchy && activeSession
      ? buildJointClusterView(activeVisualView, visualHierarchy, activeSession) ?? activeVisualView
      : activeVisualView;
    return {
      roi: selectedCell.roi,
      model: selectedCell.model,
      modelCategory: inferModelCategory(selectedCell.model),
      aggregateScore: selectedCell.score,
      rankWithinRoi: selectedCell.rankWithinRoi,
      overallScore: selectedCell.overallScore,
      evidence: evidenceView
        ? { csvColumn: evidenceView.columnName, stats: evidenceView.stats, topImages: evidenceView.topImages, bottomImages: evidenceView.bottomImages }
        : null,
      clustering: {
        clusteringMethod,
        voxelSummary: activeView?.summary ?? voxelSummary,
        voxelClusters: activeView,
        voxelClustersByRoi: views,
        visualSummary: visualView?.summary ?? activeVisualSummary,
        visualClusters: visualView,
        dreamsimSummary: activeDreamsimSummary,
      },
      sessionEvidence: sessionEvidenceView,
    };
  }

  async function handleCopy() {
    const p = payload();
    if (!p) return;
    try { await copyText(JSON.stringify(p, null, 2)); setActionMsg('Copied'); }
    catch { setActionMsg('Clipboard unavailable'); }
  }

  function handleExport() {
    const p = payload();
    if (!p) return;
    downloadJson(`${p.model}-${p.roi}.json`, p);
    setActionMsg('Exported');
  }

  /* ── Empty state ── */
  if (!selectedCell) {
    return (
      <aside className="details-panel" aria-label="Details panel">
        <h2>Choose a model &amp; region</h2>
        <div className="empty-state-card">
          <p>Select a heatmap cell.</p>
        </div>
      </aside>
    );
  }

  /* ── Selected state ── */
  return (
    <aside className={`details-panel evidence-panel ${activeTab === 'clustering' ? 'details-panel--clustering' : ''}`} aria-label="Details panel">

      {/* Header */}
      <div className="evidence-panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span className="model-tag" style={{ textTransform: 'uppercase', fontWeight: 700 }}>
            {selectedCell.roi}
          </span>
          <span className="model-tag">{inferModelCategory(selectedCell.model)}</span>
        </div>
        <h2>{selectedCell.model}</h2>
        <dl className="selected-score-strip">
          <div><dt>Alignment</dt><dd>{fmt(selectedCell.score)}</dd></div>
          <div><dt>ROI rank</dt><dd>{selectedCell.rankWithinRoi === null ? '—' : `#${selectedCell.rankWithinRoi}`}</dd></div>
          <div><dt>Overall alignment</dt><dd>{fmt(selectedCell.overallScore)}</dd></div>
        </dl>
      </div>

      <div className="explorer-actions panel-action-row" aria-label="Selection actions">
        <button type="button" onClick={handleExport}>Export JSON</button>
        <button type="button" onClick={handleCopy}>Copy metadata</button>
        <span role="status" aria-live="polite">{actionMsg}</span>
      </div>

      {/* Sub-tabs */}
      <div className="detail-subtabs" role="tablist" aria-label="Detail sections">
        <button
          type="button"
          role="tab"
          className={activeTab === 'evidence' ? 'active' : ''}
          aria-selected={activeTab === 'evidence'}
          onClick={() => setActiveTab('evidence')}
        >
          Image responses
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === 'clustering' ? 'active' : ''}
          aria-selected={activeTab === 'clustering'}
          onClick={() => setActiveTab('clustering')}
        >
          Image clusters
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === 'spatial' ? 'active' : ''}
          aria-selected={activeTab === 'spatial'}
          onClick={() => setActiveTab('spatial')}
        >
          Brain &amp; voxels
        </button>
      </div>

      {/* Evidence tab */}
      <section className="image-evidence-tab" hidden={activeTab !== 'evidence'} aria-label="Image responses">
        <div className="image-evidence-toolbar">
          <label className="evidence-control">
            Images per group
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
              {topKOptions.map((k) => <option key={k} value={k}>{k} highest / lowest</option>)}
            </select>
          </label>
        </div>
        {sessionEvidenceView ? (
          <SessionEvidencePanel
            view={sessionEvidenceView}
            onOpenImage={setModalImage}
            onFocusStimulus={onFocusStimulus}
          />
        ) : !evidenceView ? (
          <div className="evidence-empty-state">
            <h3>Image responses unavailable</h3>
            <p>This selection has alignment scores only.</p>
          </div>
        ) : (
          <>
            <p className="evidence-value-label">Images ranked by mean predicted voxel response.</p>
            <div className="evidence-tier-grid">
              <ImageGrid title="Highest predicted responses" images={evidenceView.topImages} onOpenImage={setModalImage} compact />
              <ImageGrid title="Lowest predicted responses" images={evidenceView.bottomImages} onOpenImage={setModalImage} compact />
            </div>
            <details className="evidence-source-details">
              <summary>Response details</summary>
              <p>Values are mean predicted voxel responses, not model alignment scores.</p>
              <dl className="summary-card-grid evidence-mini-stats">
                <div><dt>Highest response</dt><dd>{fmtV(evidenceView.stats.max)}</dd></div>
                <div><dt>Lowest response</dt><dd>{fmtV(evidenceView.stats.min)}</dd></div>
              </dl>
              <p className="evidence-column-note">Source column: <strong>{evidenceView.columnName}</strong></p>
            </details>
          </>
        )}
      </section>

      {/* Clustering tab */}
      <section hidden={activeTab !== 'clustering'} aria-label="Image clusters">
        <ClusteringPanel
          voxelRoiStates={voxelRoiStates}
          clusteringMethod={clusteringMethod}
          clusteringMethodOptions={CLUSTERING_METHODS}
          onClusteringMethodChange={setClusteringMethod}
          visualView={activeVisualView}
          dreamsimView={activeDreamsimView}
          imageCategories={clustering.imageCategories}
          groundTruthPatients={activeGroundTruthPatients}
          onOpenImage={setModalImage}
          session={sessionEvidenceView ? session : null}
          selectedModel={selectedCell.model}
          selectedRoi={selectedCell.roi}
          selectivityAtlasAvailable={Boolean(voxelSummary)}
          showSelectivityLauncher={false}
          focusedStimulusId={focusedStimulusId}
        />
      </section>

      <section hidden={activeTab !== 'spatial'} aria-label="Brain and voxels">
        {activeTab === 'spatial' && (
          <SpatialTools
            model={selectedCell.model}
            session={session}
            focusedStimulusId={focusedStimulusId}
            onFocusStimulus={onFocusStimulus}
          />
        )}
      </section>

      {modalImage && <ImageModal image={modalImage} onClose={() => setModalImage(null)} />}
    </aside>
  );
}
