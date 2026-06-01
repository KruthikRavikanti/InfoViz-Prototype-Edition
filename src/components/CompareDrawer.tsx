import { useEffect, useMemo, useState } from 'react';
import { ImageGrid } from './ImageGrid';
import type { AggregateHeatmapCell, EvidenceImage, ModelRoiColumn, SelectedHeatmapCell, WideCsvRow } from '../types/data';
import { downloadJson } from '../utils/browserActions';
import { buildCompareSummary } from '../utils/compare';
import { findSharedCompareNeighbors } from '../utils/discovery';
import { buildEvidenceView } from '../utils/evidence';
import { inferModelCategory } from '../utils/modelTags';

type CompareDrawerProps = {
  compareMode: boolean;
  compareCells: SelectedHeatmapCell[];
  heatmapCells: AggregateHeatmapCell[];
  rows: WideCsvRow[];
  modelRoiColumns: ModelRoiColumn[];
  topK: number;
  onCompareModeChange: (enabled: boolean) => void;
};

function fmt(score: number | null) { return score === null ? '—' : score.toFixed(3); }
function fmtDiff(v: number | null) {
  if (v === null) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
}

function ImageModal({ image, onClose }: { image: EvidenceImage; onClose: () => void }) {
  const [missing, setMissing] = useState(false);
  return (
    <div className="image-modal-backdrop" role="presentation" onClick={onClose}>
      <div className="image-modal" role="dialog" aria-modal="true" aria-label={image.imageName}
           onClick={(e) => e.stopPropagation()}>
        <div className="image-modal-header">
          <div>
            <h3>{image.imageName}</h3>
            <p>{image.valueLabel ?? `Value: ${image.value.toFixed(3)}`}</p>
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

function CellCard({
  label, cell, evidenceView, onOpenImage,
}: {
  label: 'A' | 'B';
  cell: SelectedHeatmapCell;
  evidenceView: ReturnType<typeof buildEvidenceView>;
  onOpenImage: (i: EvidenceImage) => void;
}) {
  return (
    <article className="compare-cell-card">
      <div className="compare-card-heading">
        <span>{label}</span>
        <div>
          <h3>{cell.model}</h3>
          <p>
            <span style={{ textTransform: 'uppercase', fontWeight: 700 }}>{cell.roi}</span>
            {' · '}
            <span className="model-tag">{inferModelCategory(cell.model)}</span>
          </p>
        </div>
      </div>
      <dl className="compare-metrics">
        <div><dt>Score</dt><dd>{fmt(cell.score)}</dd></div>
        <div><dt>ROI rank</dt><dd>{cell.rankWithinRoi === null ? '—' : `#${cell.rankWithinRoi}`}</dd></div>
      </dl>
      {evidenceView ? (
        <>
          <ImageGrid title={`Top ${label}`}    images={evidenceView.topImages}    onOpenImage={onOpenImage} compact />
          <ImageGrid title={`Bottom ${label}`} images={evidenceView.bottomImages} onOpenImage={onOpenImage} compact />
        </>
      ) : (
        <p className="compare-empty">No matching CSV column found.</p>
      )}
    </article>
  );
}

export function CompareDrawer({
  compareMode, compareCells, heatmapCells, rows, modelRoiColumns, topK, onCompareModeChange,
}: CompareDrawerProps) {
  const [modalImage, setModalImage] = useState<EvidenceImage | null>(null);
  const [cellA, cellB] = compareCells;

  const evidenceA = useMemo(
    () => (cellA ? buildEvidenceView(cellA, rows, modelRoiColumns, topK) : null),
    [cellA, modelRoiColumns, rows, topK],
  );
  const evidenceB = useMemo(
    () => (cellB ? buildEvidenceView(cellB, rows, modelRoiColumns, topK) : null),
    [cellB, modelRoiColumns, rows, topK],
  );
  const summary = cellA && cellB ? buildCompareSummary(cellA, evidenceA, cellB, evidenceB) : null;

  // kept for future use
  const _neighbors = useMemo(
    () => (cellA && cellB ? findSharedCompareNeighbors(cellA, evidenceA, cellB, evidenceB, heatmapCells, rows, modelRoiColumns, topK) : []),
    [cellA, cellB, evidenceA, evidenceB, heatmapCells, modelRoiColumns, rows, topK],
  );

  useEffect(() => {
    if (!cellA || !cellB) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onCompareModeChange(false); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [cellA, cellB, onCompareModeChange]);

  function handleExport() {
    if (!cellA || !cellB || !summary) return;
    downloadJson('compare-summary.json', {
      cellA: { roi: cellA.roi, model: cellA.model, modelCategory: inferModelCategory(cellA.model), aggregateScore: cellA.score, evidence: evidenceA },
      cellB: { roi: cellB.roi, model: cellB.model, modelCategory: inferModelCategory(cellB.model), aggregateScore: cellB.score, evidence: evidenceB },
      summary,
    });
  }

  if (!compareMode || !cellA || !cellB) return null;

  return (
    <div className="compare-modal-backdrop" role="presentation" onClick={() => onCompareModeChange(false)}>
      <section
        className="compare-drawer compare-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Compare two cells"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Heading */}
        <div className="compare-drawer-heading">
          <div>
            <h2>Compare</h2>
            <p>Image overlap and score differences between the two selected cells.</p>
          </div>
          <button className="compare-close-button" type="button" onClick={() => onCompareModeChange(false)}>
            Close &thinsp; <kbd style={{ fontFamily: 'inherit', fontSize: '.72rem', opacity: .7 }}>Esc</kbd>
          </button>
        </div>

        <div className="compare-drawer-body">
          <p className="compare-selection-note">{compareCells.length} / 2 cells selected</p>

          {/* Side-by-side cards */}
          <div className="compare-card-grid">
            <CellCard label="A" cell={cellA} evidenceView={evidenceA} onOpenImage={setModalImage} />
            <CellCard label="B" cell={cellB} evidenceView={evidenceB} onOpenImage={setModalImage} />
          </div>

          {/* Overlap section */}
          {summary ? (
            <section className="compare-overlap-section">
              <div className="compare-badge-row">
                <span>Top overlap: {summary.top.overlap.length}</span>
                <span>Bottom overlap: {summary.bottom.overlap.length}</span>
                <span>Score diff A–B: {fmtDiff(summary.aggregateScoreDifference)}</span>
                <span>Mean diff A–B: {fmtDiff(summary.imageMeanDifference)}</span>
              </div>

              <p className="compare-summary-note">
                {summary.top.overlap.length > 0 || summary.bottom.overlap.length > 0 ? (
                  <>
                    These cells share <strong>{summary.top.overlap.length}</strong> top images
                    and <strong>{summary.bottom.overlap.length}</strong> bottom images — useful for
                    "same evidence, different score" inspection.
                  </>
                ) : (
                  'No shared images in the current evidence window. Any aggregate similarity likely comes from different stimuli.'
                )}
              </p>

              <div className="panel-action-row">
                <button type="button" onClick={handleExport}>Export JSON</button>
              </div>

              <div className="compare-overlap-grid">
                <ImageGrid title="Overlapping top"    images={summary.top.overlap}   onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique to A (top)"  images={summary.top.uniqueA}   onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique to B (top)"  images={summary.top.uniqueB}   onOpenImage={setModalImage} compact />
              </div>
              <div className="compare-overlap-grid">
                <ImageGrid title="Overlapping bottom"    images={summary.bottom.overlap}  onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique to A (bottom)"  images={summary.bottom.uniqueA}  onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique to B (bottom)"  images={summary.bottom.uniqueB}  onOpenImage={setModalImage} compact />
              </div>
            </section>
          ) : (
            <p className="compare-empty">
              Both cells need matching CSV columns before image overlap can be computed.
            </p>
          )}
        </div>

        {modalImage && <ImageModal image={modalImage} onClose={() => setModalImage(null)} />}
      </section>
    </div>
  );
}
