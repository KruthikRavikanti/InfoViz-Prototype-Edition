import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { ClusteringPanel, VoxelRoiComparison } from './ClusteringPanel';
import { CLUSTERING_METHODS } from '../utils/clustering';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import type { ClusterPoint, ClusterView } from '../types/data';
import { jointHierarchyFixture } from '../test/jointHierarchyFixture';

const commonProps = {
  voxelRoiStates: {
    ffa: { view: null, status: 'idle' as const },
    eba: { view: null, status: 'idle' as const },
    ppa: { view: null, status: 'idle' as const },
  },
  clusteringMethod: 'kmeans_euclidean' as const,
  clusteringMethodOptions: CLUSTERING_METHODS,
  onClusteringMethodChange: vi.fn(),
  visualView: null,
  dreamsimView: null,
  imageCategories: new Map(),
  groundTruthPatients: [],
  onOpenImage: vi.fn(),
  selectedModel: 'clip_rn50',
  selectedRoi: 'ffa',
  selectivityAtlasAvailable: true,
  focusedStimulusId: 'stimulus-a',
};

const referencePoints: ClusterPoint[] = [
  {
    imageName: 'reference-a.png', imageUrl: '/images/reference-a.png', clusterLabel: '2', value: .7, rank: 1,
    pcaX: null, pcaY: null, tsneX: null, tsneY: null, plotX: 0, plotY: 0, distanceToCentroid: .1,
  },
  {
    imageName: 'reference-b.png', imageUrl: '/images/reference-b.png', clusterLabel: '2', value: .6, rank: 2,
    pcaX: null, pcaY: null, tsneX: null, tsneY: null, plotX: 2, plotY: -3, distanceToCentroid: .2,
  },
];

const populatedVoxelView: ClusterView = {
  summary: null,
  points: referencePoints,
  groups: [{
    label: '2',
    size: 2,
    points: referencePoints,
    images: referencePoints.map((point) => ({
      imageName: point.imageName,
      imageUrl: point.imageUrl,
      value: point.value,
      rank: point.rank,
    })),
  }],
  coordinateMode: 'plot',
};

const populatedVoxelRoiStates = {
  ffa: { view: populatedVoxelView, status: 'idle' as const },
  eba: { view: populatedVoxelView, status: 'idle' as const },
  ppa: { view: populatedVoxelView, status: 'idle' as const },
};

describe('session clustering safeguards', () => {
  it('does not claim measured ground-truth data for uploaded stimuli', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const markup = renderToStaticMarkup(<ClusteringPanel {...commonProps} session={session} />);

    expect(markup).toContain('data-no-upload-overlay="true"');
    expect(markup).toContain('Uploads are predictions only; this view uses measured fMRI.');
    expect(markup).toContain('data-dreamsim-overlay="disabled"');
    expect(markup).toContain('1,792 features');
  });

  it('leaves standalone clustering free of session-only notes', () => {
    const markup = renderToStaticMarkup(<ClusteringPanel {...commonProps} session={null} />);
    expect(markup).not.toContain('data-no-upload-overlay="true"');
    expect(markup).not.toContain('data-dreamsim-overlay="disabled"');
  });

  it('embeds an uploaded stimulus as a small glowing plot marker', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const markup = renderToStaticMarkup(
      <ClusteringPanel {...commonProps} voxelRoiStates={populatedVoxelRoiStates} session={session} />,
    );

    expect(markup).toContain('data-uploaded-marker="true"');
    expect(markup).toContain('session-cluster-marker-glow');
    expect(markup).toContain('session-cluster-marker-core');
    expect(markup).toContain('aria-label="Uploaded stimulus uploaded.png, cluster 2, centroid distance 0.420"');
    expect(new Set(Array.from(markup.matchAll(/data-marker-color="([^"]+)"/g), (match) => match[1]))).toEqual(new Set(['#1c1917']));
    expect(markup).not.toContain('<title>Uploaded: uploaded.png');
    expect(markup).not.toContain('M 0 -9 L 9 0 L 0 9 L -9 0 Z');
  });

  it('does not add upload markers to standalone plots', () => {
    const markup = renderToStaticMarkup(
      <ClusteringPanel {...commonProps} voxelRoiStates={populatedVoxelRoiStates} session={null} />,
    );

    expect(markup).not.toContain('data-uploaded-marker="true"');
  });

  it('uses fresh joint labels, combined counts, and a new tree without moving reference axes', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
    session.roiResults.ffa.stimuli[0].hierarchicalAssignment.clusterId = '9';
    session.roiResults.ffa.jointHierarchy = jointHierarchyFixture();
    const view: ClusterView = {
      ...populatedVoxelView,
      uploadAxisAlignment: { xSign: 1, ySign: -1 },
      summary: {
        clusteringMethod: 'hierarchical', dendrogramPng: '/fixed-dendrogram.png',
        nImages: 2, bestK: 1, silhouette: null, clusterSizes: { '2': 2 },
      },
    };
    const original = structuredClone(view);
    const markup = renderToStaticMarkup(
      <ClusteringPanel
        {...commonProps}
        clusteringMethod="hierarchical_correlation"
        voxelRoiStates={{ ...populatedVoxelRoiStates, ffa: { view, status: 'idle' } }}
        session={session}
      />,
    );

    expect(markup).toContain('data-uploaded-marker="true"');
    expect(markup).toContain('data-assignment-method="joint hierarchical clustering"');
    expect(markup).toContain('joint hierarchical cluster 1, correlation distance to cluster medoid 0.230, medoid reference-b.png');
    expect(markup).not.toContain('nearest-medoid assignment');
    expect(markup).not.toContain('centroid distance 0.420');
    expect(markup).toContain('display axes stay fixed');
    expect(markup).toContain('they do not assign membership');
    expect(markup).not.toContain('src="/fixed-dendrogram.png"');
    expect(markup.match(/data-link-key="reference:reference-a\.png"/g)).toHaveLength(1);
    expect(markup).toContain('2 clusters · 3 images (2 reference + 1 uploaded)');
    expect(view).toEqual(original);
  });

  it('leaves legacy sessions reference-only instead of silently using approximate assignments', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
    const markup = renderToStaticMarkup(
      <ClusteringPanel
        {...commonProps}
        clusteringMethod="hierarchical_correlation"
        voxelRoiStates={populatedVoxelRoiStates}
        session={session}
      />,
    );

    expect(markup).not.toContain('data-uploaded-marker="true"');
    expect(markup).toContain('This upload session predates joint clustering.');
    expect(markup).not.toContain('nearest-medoid assignment');
    expect(markup).not.toContain('Cluster 2 · correlation distance 0.230');
  });

  it('preserves linked upload markers across all three hierarchical ROI charts', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
    for (const roi of ['ffa', 'eba', 'ppa'] as const) session.roiResults[roi].jointHierarchy = jointHierarchyFixture();
    const view = { ...populatedVoxelView, uploadAxisAlignment: { xSign: 1, ySign: -1 } as const };
    const markup = renderToStaticMarkup(
      <VoxelRoiComparison
        voxelRoiStates={{
          ffa: { view, status: 'idle' }, eba: { view, status: 'idle' }, ppa: { view, status: 'idle' },
        }}
        clusteringMethod="hierarchical_correlation"
        imageCategories={new Map()}
        onOpenImage={vi.fn()}
        session={session}
        selectedRoi="ffa"
        focusedStimulusId="stimulus-a"
      />,
    );

    expect(markup.match(/data-link-key="uploaded:stimulus-a"/g)).toHaveLength(3);
    expect(markup.match(/data-assignment-method="joint hierarchical clustering"/g)).toHaveLength(3);
    expect(markup.match(/data-link-key="reference:reference-a\.png"/g)).toHaveLength(3);
    expect(markup.match(/2 clusters · 3 images/g)).toHaveLength(3);
  });

  it('keeps true joint labels and tree available when upload axis checks fail', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
    session.roiResults.ffa.jointHierarchy = jointHierarchyFixture();
    const markup = renderToStaticMarkup(
      <ClusteringPanel
        {...commonProps}
        clusteringMethod="hierarchical_correlation"
        voxelRoiStates={populatedVoxelRoiStates}
        session={session}
      />,
    );
    expect(markup).not.toContain('data-uploaded-marker="true"');
    expect(markup).toContain('reference axes could not be verified');
    expect(markup).toContain('2 clusters · 3 images');
    expect(markup).toContain('Joint hierarchical clustering');
  });

  it('does not show session markers for a different selected model', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
    const markup = renderToStaticMarkup(
      <ClusteringPanel
        {...commonProps}
        selectedModel="another_model"
        voxelRoiStates={populatedVoxelRoiStates}
        session={session}
      />,
    );
    expect(markup).not.toContain('data-uploaded-marker="true"');
    expect(markup).not.toContain('uploaded placements');
  });

  it('keeps the original four-chart board and leaves ROI comparison in a dialog launcher', () => {
    const markup = renderToStaticMarkup(
      <ClusteringPanel {...commonProps} voxelRoiStates={populatedVoxelRoiStates} session={null} />,
    );

    expect(markup).toContain('Predicted-response clusters');
    expect(markup).toContain('Visual-feature clusters');
    expect(markup).toContain('DreamSim similarity clusters');
    expect(markup).toContain('Measured fMRI clusters');
    expect(markup).toContain('Each dot is an image; colors mark clusters. Hover to preview; click to enlarge.');
    expect(markup.match(/1 cluster · 2 images/g)).toHaveLength(1);
    expect(markup).toContain('<details class="cluster-details"><summary>Cluster details</summary>');
    expect(markup).toContain('PCA projection. Axes are fixed to the reference images.');
    expect(markup).toContain('aria-label="PCA image-cluster projection"');
    expect(markup).toContain('Compare FFA · EBA · PPA');
    expect(markup).toContain('Voxel Selectivity Explorer');
    expect(markup).toContain('aria-expanded="false"');
    expect(markup).not.toContain('data-roi-cluster-chart');
    expect(markup).not.toContain('role="dialog"');
    expect(CLUSTERING_METHODS.map((method) => method.id)).toEqual([
      'kmeans_euclidean',
      'hierarchical_correlation',
    ]);
    expect(markup.match(/<option/g)).toHaveLength(2);
  });

  it('guards the selectivity explorer launcher when the selected model has no audited voxel artifact', () => {
    const markup = renderToStaticMarkup(
      <ClusteringPanel
        {...commonProps}
        selectivityAtlasAvailable={false}
        voxelRoiStates={populatedVoxelRoiStates}
        session={null}
      />,
    );

    expect(markup).toContain('Voxel Selectivity Explorer');
    expect(markup).toContain('disabled=""');
    expect(markup).toContain('Voxel selectivity is not available for this model');
  });

  it('renders linked comparison charts for FFA, EBA, and PPA inside the comparison view', () => {
    const markup = renderToStaticMarkup(
      <VoxelRoiComparison
        voxelRoiStates={populatedVoxelRoiStates}
        clusteringMethod="kmeans_euclidean"
        imageCategories={new Map()}
        onOpenImage={vi.fn()}
        session={null}
        selectedRoi="ffa"
        focusedStimulusId={null}
      />,
    );

    expect(markup).toContain('data-roi-cluster-chart="ffa"');
    expect(markup).toContain('data-roi-cluster-chart="eba"');
    expect(markup).toContain('data-roi-cluster-chart="ppa"');
    expect(markup.match(/data-link-key="reference:reference-a\.png"/g)).toHaveLength(3);
    expect(markup).toContain('Hover a point to trace the same image across FFA, EBA, and PPA.');
    expect(markup).toContain('Predicted-response clusters across ROIs');
    expect(markup).toContain('aria-label="FFA PCA image-cluster projection"');
  });

  it('keeps an out-of-range upload warning visible outside collapsible cluster details', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    session.roiResults.ffa.stimuli[0].voxelProjection.x = 10;
    const markup = renderToStaticMarkup(
      <ClusteringPanel {...commonProps} voxelRoiStates={populatedVoxelRoiStates} session={session} />,
    );

    expect(markup).toContain('<p class="cluster-boundary-note" role="note">');
    expect(markup).toContain('Uploads outside the reference range are pinned to the plot boundary.');
    expect(markup).toContain('shown at the fixed reference boundary');
    expect(markup).not.toMatch(/<figcaption>Uploads outside/);
  });
});
