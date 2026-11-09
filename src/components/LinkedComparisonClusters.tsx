import { useEffect, useState } from 'react';
import type { ClusterView, ClusteringData, ClusteringMethod, EvidenceImage, SelectedHeatmapCell } from '../types/data';
import type { SharedResponsePoint } from '../utils/responseComparison';
import { CLUSTERING_METHODS, loadVoxelClusterView } from '../utils/clustering';
import { isCanonicalRoi } from '../utils/sessionEvidence';
import { ClusterScatter } from './ClusteringPanel';
import { ComparisonPinMarker } from './ComparisonPinMarker';
import './LinkedComparisonClusters.css';

type LoadState = {
  key: string;
  status: 'loading' | 'ready' | 'error';
  views: [ClusterView | null, ClusterView | null];
};

export function pinnedClusterRelationship(view: ClusterView, pinnedImages: SharedResponsePoint[]): 'same' | 'separate' | 'incomplete' {
  if (pinnedImages.length !== 2) return 'incomplete';
  const labels = pinnedImages.map((pin) => view.points.find((p) => p.imageName === pin.imageName)?.clusterLabel);
  if (labels.some((label) => label === undefined)) return 'incomplete';
  return labels[0] === labels[1] ? 'same' : 'separate';
}

export function LinkedComparisonClusters({ cellA, cellB, clustering, pinnedImages, onOpenImage }: {
  cellA: SelectedHeatmapCell;
  cellB: SelectedHeatmapCell;
  clustering: ClusteringData;
  pinnedImages: SharedResponsePoint[];
  onOpenImage: (image: EvidenceImage) => void;
}) {
  const [method, setMethod] = useState<ClusteringMethod>('hierarchical_correlation');
  const key = `${cellA.model}\u0000${cellA.roi}\u0000${cellB.model}\u0000${cellB.roi}\u0000${method}`;
  const [state, setState] = useState<LoadState>({ key: '', status: 'loading', views: [null, null] });
  const sameRoi = cellA.roi === cellB.roi;
  const summaries = clustering.voxelSummariesByMethod[method];

  useEffect(() => {
    if (!sameRoi) return;
    let ignore = false;
    setState({ key, status: 'loading', views: [null, null] });
    Promise.all([
      loadVoxelClusterView(summaries, { model: cellA.model, roi: cellA.roi }),
      loadVoxelClusterView(summaries, { model: cellB.model, roi: cellB.roi }),
    ]).then((views) => {
      if (!ignore) setState({ key, status: 'ready', views });
    }).catch(() => {
      if (!ignore) setState({ key, status: 'error', views: [null, null] });
    });
    return () => { ignore = true; };
  }, [sameRoi, key, summaries, cellA.model, cellA.roi, cellB.model, cellB.roi]);

  if (!sameRoi) {
    return <p className="compare-empty">Choose the same ROI for both models to compare pinned image clusters.</p>;
  }
  const ready = state.key === key && state.status === 'ready';
  const [viewA, viewB] = ready ? state.views : [null, null];
  const aIds = viewA ? new Set(viewA.points.map((p) => p.imageName)) : null;
  const matchingReferences = Boolean(viewA && viewB && aIds && aIds.size === viewA.points.length
    && new Set(viewB.points.map((p) => p.imageName)).size === viewB.points.length
    && aIds.size === viewB.points.length
    && viewB.points.every((p) => aIds.has(p.imageName)));

  return (
    <section className="linked-comparison-clusters" aria-label="Pinned image cluster comparison">
      <div className="linked-cluster-toolbar">
        <div>
          <h3>Pinned images in response clusters</h3>
          <p>Same image markers in both models · {cellA.roi.toUpperCase()} · reference images only.</p>
        </div>
        <label className="cluster-method-control">
          Clustering method
          <select value={method} onChange={(event) => setMethod(event.target.value as ClusteringMethod)} aria-label="Pinned cluster comparison method">
            {CLUSTERING_METHODS.map((option) => <option key={option.id} value={option.id}>{option.label} · {option.description}</option>)}
          </select>
        </label>
      </div>
      {state.key === key && state.status === 'error' ? (
        <p role="status">Response clusters could not be loaded for this comparison.</p>
      ) : !ready ? (
        <p role="status">Loading response clusters…</p>
      ) : !viewA || !viewB ? (
        <p role="status">Response clusters are unavailable for one of these selections.</p>
      ) : !matchingReferences ? (
        <p role="status">The model clusters use different reference image sets.</p>
      ) : (
        <>
          <p className="linked-cluster-reference-count">{viewA.points.length} shared reference images · colors identify each model’s clusters; compare membership, not cluster numbers.</p>
          {pinnedImages.length < 2 && <p className="linked-cluster-pin-instruction">Pin two images in the response scatterplot to follow their cluster memberships.</p>}
          <div className="linked-cluster-model-grid">
            {[{ cell: cellA, view: viewA, label: 'A', rankKey: 'rankA' as const }, { cell: cellB, view: viewB, label: 'B', rankKey: 'rankB' as const }].map(({ cell, view, label, rankKey }) => {
              const relationship = pinnedClusterRelationship(view, pinnedImages);
              const pins = pinnedImages.map((pin, index) => ({ pin, index, point: view.points.find((p) => p.imageName === pin.imageName) }));
              return (
                <article className="linked-cluster-model-card" key={`${cell.model}-${cell.roi}`} data-comparison-model={cell.model}>
                  <header>
                    <h4>{label} · {cell.model}</h4>
                    <span>{cell.roi.toUpperCase()} · {view.groups.length} clusters · {view.points.length} images</span>
                  </header>
                  <p className="linked-cluster-pair-result" data-pair-relationship={relationship}>
                    {relationship === 'same' ? 'Pinned images are in the same cluster.'
                      : relationship === 'separate' ? 'Pinned images are in separate clusters.'
                        : pinnedImages.length === 2 ? 'A pinned image is absent from this cluster view.' : 'Select two pinned images to compare.'}
                  </p>
                  <ClusterScatter
                    view={view}
                    onOpenImage={onOpenImage}
                    searchQuery=""
                    categoryHighlight={null}
                    imageCategories={clustering.imageCategories}
                    variant="full"
                    roi={isCanonicalRoi(cell.roi) ? cell.roi : undefined}
                    pinnedImages={pins.map(({ pin, index }) => ({ imageName: pin.imageName, index }))}
                  />
                  <div className="linked-cluster-size-list" aria-label={`${cell.model} cluster sizes`}>
                    {view.groups.map((group) => <span key={group.label}>Cluster {group.label}: <strong>{group.size}</strong></span>)}
                  </div>
                  <div className="linked-cluster-pin-cards">
                    {pins.map(({ pin, index, point }) => (
                      <button
                        key={pin.imageId} type="button" className="linked-cluster-pin-card"
                        aria-label={`Pinned image ${index + 1}, ${pin.imageName}, ${point ? `cluster ${point.clusterLabel}` : 'not mapped'}`}
                        onClick={() => onOpenImage({ imageName: pin.imageName, imageUrl: pin.imageUrl, value: rankKey === 'rankA' ? pin.meanA : pin.meanB, rank: pin[rankKey], valueLabel: point ? `Cluster ${point.clusterLabel} · rank #${pin[rankKey]}` : 'Not present in this clustering' })}
                      >
                        <img src={pin.imageUrl} alt="" />
                        <span>
                          <strong><svg viewBox="0 0 28 28" aria-hidden="true"><ComparisonPinMarker index={index} x={14} y={14} /></svg>{index + 1} · {pin.imageId}</strong>
                          <span>{point ? `Cluster ${point.clusterLabel}` : 'Not present'} · rank #{pin[rankKey]}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                  <p className="linked-cluster-model-note">{[
                    view.summary?.linkageMethod && `${view.summary.linkageMethod} linkage`,
                    view.summary?.normalization === 'voxel_zscore' ? 'voxel z-score' : view.summary?.normalization,
                    view.summary?.nVoxels && `${view.summary.nVoxels} voxels`,
                  ].filter(Boolean).join(' · ')}</p>
                </article>
              );
            })}
          </div>
          <p className="linked-cluster-axis-note">Positions are independently fitted PCA projections; cluster membership uses full predicted-response patterns.</p>
        </>
      )}
    </section>
  );
}
