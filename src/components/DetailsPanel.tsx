import { useEffect, useMemo, useState } from 'react';
import { ClusteringPanel } from './ClusteringPanel';
import { ImageGrid } from './ImageGrid';
import type { AggregateHeatmapCell, ClusteringData, ClusterView, EvidenceImage, ModelRoiColumn, SelectedHeatmapCell, WideCsvRow } from '../types/data';
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

function formatScore(score: number | null): string {
  return score === null ? '—' : score.toFixed(3);
}

function formatValue(value: number): string {
  return value.toFixed(3);
}

function roiChipClass(roi: string): string {
  const lower = roi.toLowerCase();
  if (lower === 'ffa' || lower === 'ppa' || lower === 'eba') return `roi-chip ${lower}`;
  return 'model-tag';
}

function ImageModal({ image, onClose }: { image: EvidenceImage; onClose: () => void }) {
  const [imageMissing, setImageMissing] = useState(false);

  return (
    <div className="image-modal-backdrop" role="presentation" onClick={onClose}>
      <div className="image-modal" role="dialog" aria-modal="true" aria-label={image.imageName} onClick={(event) => event.stopPropagation()}>
        <div className="image-modal-header">
          <div>
            <h3>{image.imageName}</h3>
            <p>{image.valueLabel ?? `Value: ${formatValue(image.value)}`}</p>
          </div>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
        {imageMissing ? (
          <span className="modal-missing-image">Image unavailable</span>
        ) : (
          <img src={image.imageUrl} alt={image.imageName} onError={() => setImageMissing(true)} />
        )}
      </div>
    </div>
  );
}

export function DetailsPanel({
  heatmapCells,
  imageCount,
  clustering,
  modelRoiColumns,
  onSelectCell,
  rows,
  selectedCell,
}: DetailsPanelProps) {
  const [topK, setTopK] = useState(6);
  const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>('evidence');
  const [modalImage, setModalImage] = useState<EvidenceImage | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [voxelClusterView, setVoxelClusterView] = useState<ClusterView | null>(null);
  const [voxelClusterStatus, setVoxelClusterStatus] = useState<'idle' | 'loading' | 'error'>('idle');

  const evidenceView = useMemo(() => {
    if (!selectedCell) {
      return null;
    }

    return buildEvidenceView(selectedCell, rows, modelRoiColumns, topK);
  }, [modelRoiColumns, rows, selectedCell, topK]);

  const voxelSummary = useMemo(() => {
    if (!selectedCell) {
      return null;
    }

    return findVoxelClusterSummary(clustering.voxelSummaries, selectedCell);
  }, [clustering.voxelSummaries, selectedCell]);

  useEffect(() => {
    let ignore = false;

    if (!selectedCell || !voxelSummary) {
      setVoxelClusterView(null);
      setVoxelClusterStatus('idle');
      return () => {
        ignore = true;
      };
    }

    setVoxelClusterStatus('loading');
    setVoxelClusterView(null);

    loadVoxelClusterView(clustering.voxelSummaries, selectedCell)
      .then((view) => {
        if (!ignore) {
          setVoxelClusterView(view);
          setVoxelClusterStatus('idle');
        }
      })
      .catch(() => {
        if (!ignore) {
          setVoxelClusterView(null);
          setVoxelClusterStatus('error');
        }
      });

    return () => {
      ignore = true;
    };
  }, [clustering.voxelSummaries, selectedCell, voxelSummary]);

  function selectedPayload() {
    if (!selectedCell) {
      return null;
    }

    return {
      roi: selectedCell.roi,
      model: selectedCell.model,
      modelCategory: inferModelCategory(selectedCell.model),
      aggregateScore: selectedCell.score,
      rankWithinRoi: selectedCell.rankWithinRoi,
      overallScore: selectedCell.overallScore,
      evidence: evidenceView
        ? {
            csvColumn: evidenceView.columnName,
            stats: evidenceView.stats,
            topImages: evidenceView.topImages,
            bottomImages: evidenceView.bottomImages,
          }
        : null,
      clustering: {
        voxelSummary,
        voxelClusters: voxelClusterView,
        visualSummary: clustering.visualSummary,
      },
    };
  }

  async function handleCopyMetadata() {
    const payload = selectedPayload();

    if (!payload) {
      return;
    }

    try {
      await copyText(JSON.stringify(payload, null, 2));
      setActionMessage('Copied to clipboard');
    } catch {
      setActionMessage('Clipboard unavailable');
    }
  }

  function handleExportSelection() {
    const payload = selectedPayload();

    if (!payload) {
      return;
    }

    downloadJson(`selected-${payload.model}-${payload.roi}.json`, payload);
    setActionMessage('Exported JSON');
  }

  if (!selectedCell) {
    return (
      <aside className="details-panel" aria-label="Details panel">
        <p className="eyebrow">Details</p>
        <h2>Cell Inspector</h2>
        <p style={{ marginTop: 8, marginBottom: 16, fontSize: '0.82rem' }}>
          Click a heatmap cell to rank images by the matching CSV response column.
        </p>
        <div className="empty-state-card">
          <h3>No cell selected</h3>
          <p>Use the heatmap to choose one ROI/model cell, or enable compare mode to select two cells.</p>
        </div>
        <dl className="details-list">
          <div>
            <dt>Selection</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Loaded images</dt>
            <dd>{imageCount}</dd>
          </div>
        </dl>
      </aside>
    );
  }

  return (
    <aside className="details-panel evidence-panel" aria-label="Details panel">
      <div className="evidence-panel-header">
        <p className="eyebrow">Selected cell</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span className={roiChipClass(selectedCell.roi)}>{selectedCell.roi.toUpperCase()}</span>
          <span className="model-tag">{inferModelCategory(selectedCell.model)}</span>
        </div>
        <h2>{selectedCell.model}</h2>
        <dl className="selected-score-strip">
          <div>
            <dt>Score</dt>
            <dd>{formatScore(selectedCell.score)}</dd>
          </div>
          <div>
            <dt>ROI rank</dt>
            <dd>{selectedCell.rankWithinRoi === null ? '—' : `#${selectedCell.rankWithinRoi}`}</dd>
          </div>
          <div>
            <dt>Overall</dt>
            <dd>{formatScore(selectedCell.overallScore)}</dd>
          </div>
        </dl>
      </div>

      <div className="panel-action-row" aria-live="polite">
        <button type="button" onClick={handleExportSelection}>
          Export JSON
        </button>
        <button type="button" onClick={handleCopyMetadata}>
          Copy metadata
        </button>
        {actionMessage && <span>{actionMessage}</span>}
      </div>

      <label className="evidence-control">
        Images per tier
        <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
          {topKOptions.map((option) => (
            <option key={option} value={option}>
              Top / bottom {option}
            </option>
          ))}
        </select>
      </label>

      <div className="detail-subtabs" role="tablist" aria-label="Detail sections">
        <button
          type="button"
          className={activeDetailTab === 'evidence' ? 'active' : ''}
          role="tab"
          aria-selected={activeDetailTab === 'evidence'}
          onClick={() => setActiveDetailTab('evidence')}
        >
          Image evidence
        </button>
        <button
          type="button"
          className={activeDetailTab === 'clustering' ? 'active' : ''}
          role="tab"
          aria-selected={activeDetailTab === 'clustering'}
          onClick={() => setActiveDetailTab('clustering')}
        >
          Clustering
        </button>
      </div>

      <section hidden={activeDetailTab !== 'evidence'} aria-label="Image evidence">
        {!evidenceView ? (
          <div className="evidence-empty-state">
            <h3>No image-level evidence</h3>
            <p>
              The aggregate JSON contains this score, but the CSV has no matching column for{' '}
              <strong>
                {selectedCell.model}_{selectedCell.roi}
              </strong>
              .
            </p>
          </div>
        ) : (
          <>
            <dl className="summary-card-grid" style={{ marginBottom: 12 }}>
              <div>
                <dt>Max value</dt>
                <dd>{formatValue(evidenceView.stats.max)}</dd>
              </div>
              <div>
                <dt>Min value</dt>
                <dd>{formatValue(evidenceView.stats.min)}</dd>
              </div>
            </dl>

            <p className="evidence-column-note" style={{ marginBottom: 12 }}>
              Column: <strong>{evidenceView.columnName}</strong>
            </p>

            <ImageGrid title="Top images" images={evidenceView.topImages} onOpenImage={setModalImage} />
            <ImageGrid title="Bottom images" images={evidenceView.bottomImages} onOpenImage={setModalImage} />
          </>
        )}
      </section>

      <section hidden={activeDetailTab !== 'clustering'} aria-label="Clustering evidence">
        <ClusteringPanel
          voxelView={voxelClusterView}
          voxelStatus={voxelClusterStatus}
          visualView={clustering.visualView}
          onOpenImage={setModalImage}
        />
      </section>

      {modalImage && <ImageModal image={modalImage} onClose={() => setModalImage(null)} />}
    </aside>
  );
}
