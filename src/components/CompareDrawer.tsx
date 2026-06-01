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

function formatScore(score: number | null): string {
  return score === null ? '—' : score.toFixed(3);
}

function formatDifference(value: number | null): string {
  if (value === null) {
    return '—';
  }

  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

function ImageModal({ image, onClose }: { image: EvidenceImage; onClose: () => void }) {
  const [imageMissing, setImageMissing] = useState(false);

  return (
    <div className="image-modal-backdrop" role="presentation" onClick={onClose}>
      <div className="image-modal" role="dialog" aria-modal="true" aria-label={image.imageName} onClick={(event) => event.stopPropagation()}>
        <div className="image-modal-header">
          <div>
            <h3>{image.imageName}</h3>
            <p>{image.valueLabel ?? `Value: ${image.value.toFixed(3)}`}</p>
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

function CompareCellCard({
  label,
  cell,
  onOpenImage,
  evidenceView,
}: {
  label: 'A' | 'B';
  cell: SelectedHeatmapCell;
  onOpenImage: (image: EvidenceImage) => void;
  evidenceView: ReturnType<typeof buildEvidenceView>;
}) {
  return (
    <article className="compare-cell-card">
      <div className="compare-card-heading">
        <span>{label}</span>
        <div>
          <h3>{cell.model}</h3>
          <p>
            {cell.roi.toUpperCase()} &middot; <span className="model-tag">{inferModelCategory(cell.model)}</span>
          </p>
        </div>
      </div>
      <dl className="compare-metrics">
        <div>
          <dt>Aggregate</dt>
          <dd>{formatScore(cell.score)}</dd>
        </div>
        <div>
          <dt>ROI rank</dt>
          <dd>{cell.rankWithinRoi === null ? '—' : `#${cell.rankWithinRoi}`}</dd>
        </div>
      </dl>
      {evidenceView ? (
        <>
          <ImageGrid title={`Top ${label}`} images={evidenceView.topImages} onOpenImage={onOpenImage} compact />
          <ImageGrid title={`Bottom ${label}`} images={evidenceView.bottomImages} onOpenImage={onOpenImage} compact />
        </>
      ) : (
        <p className="compare-empty">No matching CSV column found.</p>
      )}
    </article>
  );
}

export function CompareDrawer({
  compareMode,
  compareCells,
  heatmapCells,
  rows,
  modelRoiColumns,
  topK,
  onCompareModeChange,
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
  const compareSummary = cellA && cellB ? buildCompareSummary(cellA, evidenceA, cellB, evidenceB) : null;
  const similarNeighbors = useMemo(
    () =>
      cellA && cellB
        ? findSharedCompareNeighbors(cellA, evidenceA, cellB, evidenceB, heatmapCells, rows, modelRoiColumns, topK)
        : [],
    [cellA, cellB, evidenceA, evidenceB, heatmapCells, modelRoiColumns, rows, topK],
  );

  // suppress unused warning — kept for future feature
  void similarNeighbors;

  useEffect(() => {
    if (!cellA || !cellB) {
      return;
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onCompareModeChange(false);
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [cellA, cellB, onCompareModeChange]);

  function handleExportCompare() {
    if (!cellA || !cellB || !compareSummary) {
      return;
    }

    downloadJson('compare-summary.json', {
      cellA: {
        roi: cellA.roi,
        model: cellA.model,
        modelCategory: inferModelCategory(cellA.model),
        aggregateScore: cellA.score,
        evidence: evidenceA,
      },
      cellB: {
        roi: cellB.roi,
        model: cellB.model,
        modelCategory: inferModelCategory(cellB.model),
        aggregateScore: cellB.score,
        evidence: evidenceB,
      },
      summary: compareSummary,
    });
  }

  if (!compareMode || !cellA || !cellB) {
    return null;
  }

  return (
    <div className="compare-modal-backdrop" role="presentation" onClick={() => onCompareModeChange(false)}>
      <section
        className="compare-drawer compare-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Compare two cells"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="compare-drawer-heading">
          <div>
            <h2>Compare mode</h2>
            <p>Image-level overlap and score differences for the two selected cells.</p>
          </div>
          <button className="compare-close-button" type="button" onClick={() => onCompareModeChange(false)}>
            Close &nbsp;Esc
          </button>
        </div>

        <div className="compare-drawer-body">
          <p className="compare-selection-note">{compareCells.length}/2 cells selected</p>

          <div className="compare-card-grid">
            <CompareCellCard
              label="A"
              cell={cellA}
              onOpenImage={setModalImage}
              evidenceView={evidenceA}
            />
            <CompareCellCard
              label="B"
              cell={cellB}
              onOpenImage={setModalImage}
              evidenceView={evidenceB}
            />
          </div>

          {compareSummary ? (
            <section className="compare-overlap-section">
              <div className="compare-badge-row">
                <span>Top overlap: {compareSummary.top.overlap.length}</span>
                <span>Bottom overlap: {compareSummary.bottom.overlap.length}</span>
                <span>Score diff A–B: {formatDifference(compareSummary.aggregateScoreDifference)}</span>
                <span>Mean diff A–B: {formatDifference(compareSummary.imageMeanDifference)}</span>
              </div>

              <p className="compare-summary-note">
                {compareSummary.top.overlap.length > 0 || compareSummary.bottom.overlap.length > 0 ? (
                  <>
                    Selections share <strong style={{ color: 'var(--compare)' }}>{compareSummary.top.overlap.length}</strong> top
                    and <strong style={{ color: 'var(--compare)' }}>{compareSummary.bottom.overlap.length}</strong> bottom images,
                    making them useful "same evidence, different score" candidates.
                  </>
                ) : (
                  <>
                    No shared top or bottom images in the current evidence window — aggregate similarity likely comes from different stimuli.
                  </>
                )}
              </p>

              <div className="panel-action-row">
                <button type="button" onClick={handleExportCompare}>
                  Export compare JSON
                </button>
              </div>

              <div className="compare-overlap-grid">
                <ImageGrid title="Overlapping top" images={compareSummary.top.overlap} onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique top A" images={compareSummary.top.uniqueA} onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique top B" images={compareSummary.top.uniqueB} onOpenImage={setModalImage} compact />
              </div>
              <div className="compare-overlap-grid">
                <ImageGrid title="Overlapping bottom" images={compareSummary.bottom.overlap} onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique bottom A" images={compareSummary.bottom.uniqueA} onOpenImage={setModalImage} compact />
                <ImageGrid title="Unique bottom B" images={compareSummary.bottom.uniqueB} onOpenImage={setModalImage} compact />
              </div>
            </section>
          ) : (
            <p className="compare-empty">Both selected cells need matching CSV columns before overlaps can be computed.</p>
          )}
        </div>

        {modalImage && <ImageModal image={modalImage} onClose={() => setModalImage(null)} />}
      </section>
    </div>
  );
}
