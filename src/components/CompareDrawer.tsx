import { Fragment, useEffect, useMemo, useState } from 'react';
import { ImageGrid } from './ImageGrid';
import type {
  AggregateHeatmapCell,
  ClusterComparisonSummary,
  ClusteringData,
  ClusteringMethod,
  EvidenceImage,
  ModelRoiColumn,
  SelectedHeatmapCell,
  WideCsvRow,
} from '../types/data';
import { downloadJson } from '../utils/browserActions';
import { calculateClusterComparison, buildCompareSummary } from '../utils/compare';
import { CLUSTERING_METHODS, loadVoxelClusterView } from '../utils/clustering';
import { findSharedCompareNeighbors } from '../utils/discovery';
import { buildEvidenceView } from '../utils/evidence';
import { inferModelCategory } from '../utils/modelTags';

type CompareDrawerProps = {
  compareMode: boolean;
  compareCells: SelectedHeatmapCell[];
  clustering: ClusteringData;
  heatmapCells: AggregateHeatmapCell[];
  rows: WideCsvRow[];
  modelRoiColumns: ModelRoiColumn[];
  topK: number;
  onCompareModeChange: (enabled: boolean) => void;
};

type ClusterComparisonStatus = 'idle' | 'loading' | 'ready' | 'error';
type ClusterComparisonsByMethod = Partial<Record<ClusteringMethod, ClusterComparisonSummary | null>>;

function fmt(score: number | null) { return score === null ? '—' : score.toFixed(3); }
function fmtDiff(v: number | null) {
  if (v === null) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
}
function fmtPercent(v: number | null) { return v === null ? '—' : `${v.toFixed(1)}%`; }
function fmtRatio(v: number | null) { return v === null ? '—' : `${(v * 100).toFixed(1)}%`; }

function sortClusterLabels(labels: string[]): string[] {
  return [...labels].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
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

function ClusterOverlapMatrix({ comparison }: { comparison: ClusterComparisonSummary }) {
  const aLabels = sortClusterLabels(Array.from(new Set(comparison.intersections.map((intersection) => intersection.aLabel))));
  const bLabels = sortClusterLabels(Array.from(new Set(comparison.intersections.map((intersection) => intersection.bLabel))));
  const countByPair = new Map(comparison.intersections.map((intersection) => [`${intersection.aLabel}\u0000${intersection.bLabel}`, intersection.count]));
  const maxCount = Math.max(...comparison.intersections.map((intersection) => intersection.count), 1);

  return (
    <div className="compare-cluster-matrix-wrap">
      <div
        className="compare-cluster-matrix"
        style={{ gridTemplateColumns: `minmax(44px,.8fr) repeat(${bLabels.length}, minmax(42px,1fr))` }}
        role="img"
        aria-label="Cluster overlap heatmap"
      >
        <span className="compare-cluster-matrix-corner">A / B</span>
        {bLabels.map((label) => (
          <span className="compare-cluster-matrix-label" key={`b-${label}`}>B{label}</span>
        ))}
        {aLabels.map((aLabel) => (
          <Fragment key={`a-row-${aLabel}`}>
            <span className="compare-cluster-matrix-label compare-cluster-matrix-label-row">A{aLabel}</span>
            {bLabels.map((bLabel) => {
              const count = countByPair.get(`${aLabel}\u0000${bLabel}`) ?? 0;
              const intensity = count / maxCount;
              return (
                <span
                  className="compare-cluster-matrix-cell"
                  key={`${aLabel}-${bLabel}`}
                  style={{ backgroundColor: `rgba(42,125,91,${0.08 + intensity * 0.48})` }}
                  title={`A${aLabel} / B${bLabel}: ${count} images`}
                >
                  {count}
                </span>
              );
            })}
          </Fragment>
        ))}
      </div>
      <p className="compare-cluster-matrix-caption">Darker cells mean more images share that A/B cluster pairing.</p>
    </div>
  );
}

function ClusterComparisonSection({
  status,
  comparisons,
}: {
  status: ClusterComparisonStatus;
  comparisons: ClusterComparisonsByMethod;
}) {
  const availableComparisons = CLUSTERING_METHODS
    .map((method) => ({ method, comparison: comparisons[method.id] ?? null }))
    .filter(({ comparison }) => comparison !== null);

  if (status === 'loading') {
    return (
      <section className="compare-cluster-section">
        <div>
          <h3>Voxel clustering similarity</h3>
          <p>Loading cluster assignments…</p>
        </div>
      </section>
    );
  }

  if (status === 'error' || availableComparisons.length === 0) {
    return (
      <section className="compare-cluster-section">
        <div>
          <h3>Voxel clustering similarity</h3>
          <p>No cluster assignments were available for both selected cells.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="compare-cluster-section">
      <div>
        <h3>Voxel clustering similarity</h3>
        <p>Labels are compared as partitions, so cluster names do not need to match.</p>
      </div>

      <div className="compare-cluster-grid">
        {availableComparisons.map(({ method, comparison }) => {
          if (!comparison) return null;
          return (
            <article className="compare-cluster-card" key={method.id}>
              <div className="compare-cluster-card-heading">
                <h4>{method.label}</h4>
                <span>{method.description}</span>
              </div>
              <dl className="compare-cluster-metrics">
                <div><dt>ARI</dt><dd>{fmt(comparison.adjustedRandIndex)}</dd></div>
                <div><dt>NMI</dt><dd>{fmt(comparison.normalizedMutualInformation)}</dd></div>
                <div><dt>Pair agreement</dt><dd>{fmtRatio(comparison.pairAgreement)}</dd></div>
                <div><dt>Co-cluster Jaccard</dt><dd>{fmtRatio(comparison.coClusterJaccard)}</dd></div>
                <div><dt>Images</dt><dd>{comparison.sharedImageCount}</dd></div>
                <div><dt>Clusters A/B</dt><dd>{comparison.clusterCountA} / {comparison.clusterCountB}</dd></div>
              </dl>
              <ClusterOverlapMatrix comparison={comparison} />
              <div className="compare-cluster-intersections">
                {comparison.largestIntersections.map((intersection) => (
                  <span key={`${intersection.aLabel}-${intersection.bLabel}`}>
                    A{intersection.aLabel} / B{intersection.bLabel}: {intersection.count}
                  </span>
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function CompareDrawer({
  compareMode, compareCells, clustering, heatmapCells, rows, modelRoiColumns, topK, onCompareModeChange,
}: CompareDrawerProps) {
  const [modalImage, setModalImage] = useState<EvidenceImage | null>(null);
  const [clusterComparisonStatus, setClusterComparisonStatus] = useState<ClusterComparisonStatus>('idle');
  const [clusterComparisons, setClusterComparisons] = useState<ClusterComparisonsByMethod>({});
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

  useEffect(() => {
    let ignore = false;

    if (!compareMode || !cellA || !cellB) {
      setClusterComparisons({});
      setClusterComparisonStatus('idle');
      return () => { ignore = true; };
    }

    setClusterComparisonStatus('loading');
    setClusterComparisons({});

    Promise.all(
      CLUSTERING_METHODS.map(async (method) => {
        try {
          const summaries = clustering.voxelSummariesByMethod[method.id] ?? [];
          const [viewA, viewB] = await Promise.all([
            loadVoxelClusterView(summaries, cellA),
            loadVoxelClusterView(summaries, cellB),
          ]);

          return [
            method.id,
            viewA && viewB ? calculateClusterComparison(viewA.points, viewB.points) : null,
          ] as const;
        } catch {
          return [method.id, null] as const;
        }
      }),
    )
      .then((entries) => {
        if (ignore) return;
        setClusterComparisons(Object.fromEntries(entries) as ClusterComparisonsByMethod);
        setClusterComparisonStatus('ready');
      })
      .catch(() => {
        if (ignore) return;
        setClusterComparisons({});
        setClusterComparisonStatus('error');
      });

    return () => { ignore = true; };
  }, [cellA, cellB, clustering, compareMode]);

  function handleExport() {
    if (!cellA || !cellB || !summary) return;
    downloadJson('compare-summary.json', {
      cellA: { roi: cellA.roi, model: cellA.model, modelCategory: inferModelCategory(cellA.model), aggregateScore: cellA.score, evidence: evidenceA },
      cellB: { roi: cellB.roi, model: cellB.model, modelCategory: inferModelCategory(cellB.model), aggregateScore: cellB.score, evidence: evidenceB },
      summary,
      clusterComparisons,
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
            <p>Image overlap, score differences, and rank similarity between the two selected cells.</p>
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
                <span className="compare-rank-similarity">
                  Rank similarity: {fmtPercent(summary.rankSimilarity.similarityScore)}
                </span>
                <span>Spearman rho: {fmt(summary.rankSimilarity.spearmanRho)}</span>
                <span>Images ranked: {summary.rankSimilarity.sharedImageCount}</span>
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

              <ClusterComparisonSection status={clusterComparisonStatus} comparisons={clusterComparisons} />

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
