import { useEffect, useMemo, useState } from 'react';
import { ClusteringPanel } from './ClusteringPanel';
import { ImageGrid } from './ImageGrid';
import type {
  AggregateHeatmapCell,
  ClusteringData,
  ClusterView,
  EvidenceImage,
  ModelRoiColumn,
  SelectedHeatmapCell,
  WideCsvRow,
} from '../types/data';
import { copyText, downloadJson } from '../utils/browserActions';
import { findVoxelClusterSummary, loadVoxelClusterView } from '../utils/clustering';
import { buildEvidenceView } from '../utils/evidence';
import { inferModelCategory } from '../utils/modelTags';

type DetailsPanelProps = {
  heatmapCells: AggregateHeatmapCell[];
  imageCount: number;
  clustering: ClusteringData;
  modelRoiColumns: ModelRoiColumn[];
  onSelectCell: (cell: SelectedHeatmapCell) => void;
  rows: WideCsvRow[];
  selectedCell: SelectedHeatmapCell | null;
};

const topKOptions = [6, 9, 12];
type DetailTab = 'evidence' | 'clustering';

function fmt(score: number | null) { return score === null ? '—' : score.toFixed(3); }
function fmtV(v: number)           { return v.toFixed(3); }

function ImageModal({ image, onClose }: { image: EvidenceImage; onClose: () => void }) {
  const [missing, setMissing] = useState(false);
  return (
    <div className="image-modal-backdrop" role="presentation" onClick={onClose}>
      <div className="image-modal" role="dialog" aria-modal="true" aria-label={image.imageName}
           onClick={(e) => e.stopPropagation()}>
        <div className="image-modal-header">
          <div>
            <h3>{image.imageName}</h3>
            <p>{image.valueLabel ?? `Value: ${fmtV(image.value)}`}</p>
          </div>
          <button type="button" onClick={onClose}>Close</button>
        </div>
        {missing
          ? <span className="modal-missing-image">Image unavailable</span>
          : <img src={image.imageUrl} alt={image.imageName} onError={() => setMissing(true)} />}
      </div>
    </div>
  );
}

export function DetailsPanel({
  heatmapCells: _heatmapCells,
  imageCount,
  clustering,
  modelRoiColumns,
  onSelectCell: _onSelectCell,
  rows,
  selectedCell,
}: DetailsPanelProps) {
  const [topK, setTopK] = useState(6);
  const [activeTab, setActiveTab] = useState<DetailTab>('evidence');
  const [modalImage, setModalImage] = useState<EvidenceImage | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const [voxelView, setVoxelView] = useState<ClusterView | null>(null);
  const [voxelStatus, setVoxelStatus] = useState<'idle' | 'loading' | 'error'>('idle');

  const evidenceView = useMemo(
    () => (selectedCell ? buildEvidenceView(selectedCell, rows, modelRoiColumns, topK) : null),
    [modelRoiColumns, rows, selectedCell, topK],
  );

  const voxelSummary = useMemo(
    () => (selectedCell ? findVoxelClusterSummary(clustering.voxelSummaries, selectedCell) : null),
    [clustering.voxelSummaries, selectedCell],
  );

  useEffect(() => {
    let ignore = false;
    if (!selectedCell || !voxelSummary) {
      setVoxelView(null);
      setVoxelStatus('idle');
      return () => { ignore = true; };
    }
    setVoxelStatus('loading');
    setVoxelView(null);
    loadVoxelClusterView(clustering.voxelSummaries, selectedCell)
      .then((v) => { if (!ignore) { setVoxelView(v); setVoxelStatus('idle'); } })
      .catch(() => { if (!ignore) { setVoxelView(null); setVoxelStatus('error'); } });
    return () => { ignore = true; };
  }, [clustering.voxelSummaries, selectedCell, voxelSummary]);

  function payload() {
    if (!selectedCell) return null;
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
      clustering: { voxelSummary, voxelClusters: voxelView, visualSummary: clustering.visualSummary },
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
        <p className="eyebrow">Inspector</p>
        <h2>Cell details</h2>
        <p style={{ marginTop: 8, marginBottom: 14, fontSize: '.82rem' }}>
          Click a heatmap cell to see image-level evidence ranked by response strength.
        </p>
        <div className="empty-state-card">
          <h3>Nothing selected</h3>
          <p>Pick any cell in the heatmap, or enable compare mode to select two.</p>
        </div>
        <dl className="details-list">
          <div><dt>Selection</dt><dd>None</dd></div>
          <div><dt>Loaded images</dt><dd>{imageCount}</dd></div>
        </dl>
      </aside>
    );
  }

  /* ── Selected state ── */
  return (
    <aside className="details-panel evidence-panel" aria-label="Details panel">

      {/* Header */}
      <div className="evidence-panel-header">
        <p className="eyebrow">Selected cell</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span className="model-tag" style={{ textTransform: 'uppercase', fontWeight: 700 }}>
            {selectedCell.roi}
          </span>
          <span className="model-tag">{inferModelCategory(selectedCell.model)}</span>
        </div>
        <h2>{selectedCell.model}</h2>
        <dl className="selected-score-strip">
          <div><dt>Score</dt><dd>{fmt(selectedCell.score)}</dd></div>
          <div><dt>ROI rank</dt><dd>{selectedCell.rankWithinRoi === null ? '—' : `#${selectedCell.rankWithinRoi}`}</dd></div>
          <div><dt>Overall</dt><dd>{fmt(selectedCell.overallScore)}</dd></div>
        </dl>
      </div>

      {/* Actions */}
      <div className="panel-action-row" aria-live="polite">
        <button type="button" onClick={handleExport}>Export JSON</button>
        <button type="button" onClick={handleCopy}>Copy metadata</button>
        {actionMsg && <span>{actionMsg}</span>}
      </div>

      {/* Top-k selector */}
      <label className="evidence-control">
        Images per tier
        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
          {topKOptions.map((k) => <option key={k} value={k}>Top / bottom {k}</option>)}
        </select>
      </label>

      {/* Sub-tabs */}
      <div className="detail-subtabs" role="tablist" aria-label="Detail sections">
        <button
          type="button"
          role="tab"
          className={activeTab === 'evidence' ? 'active' : ''}
          aria-selected={activeTab === 'evidence'}
          onClick={() => setActiveTab('evidence')}
        >
          Image evidence
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === 'clustering' ? 'active' : ''}
          aria-selected={activeTab === 'clustering'}
          onClick={() => setActiveTab('clustering')}
        >
          Clustering
        </button>
      </div>

      {/* Evidence tab */}
      <section hidden={activeTab !== 'evidence'} aria-label="Image evidence">
        {!evidenceView ? (
          <div className="evidence-empty-state">
            <h3>No image-level data</h3>
            <p>
              No CSV column found for <strong>{selectedCell.model}_{selectedCell.roi}</strong>.
            </p>
          </div>
        ) : (
          <>
            <dl className="summary-card-grid" style={{ marginBottom: 10 }}>
              <div><dt>Max value</dt><dd>{fmtV(evidenceView.stats.max)}</dd></div>
              <div><dt>Min value</dt><dd>{fmtV(evidenceView.stats.min)}</dd></div>
            </dl>
            <p className="evidence-column-note" style={{ marginBottom: 10 }}>
              Column: <strong>{evidenceView.columnName}</strong>
            </p>
            <ImageGrid title="Top images"    images={evidenceView.topImages}    onOpenImage={setModalImage} />
            <ImageGrid title="Bottom images" images={evidenceView.bottomImages} onOpenImage={setModalImage} />
          </>
        )}
      </section>

      {/* Clustering tab */}
      <section hidden={activeTab !== 'clustering'} aria-label="Clustering">
        <ClusteringPanel
          voxelView={voxelView}
          voxelStatus={voxelStatus}
          visualView={clustering.visualView}
          imageCategories={clustering.imageCategories}
          onOpenImage={setModalImage}
        />
      </section>

      {modalImage && <ImageModal image={modalImage} onClose={() => setModalImage(null)} />}
    </aside>
  );
}
