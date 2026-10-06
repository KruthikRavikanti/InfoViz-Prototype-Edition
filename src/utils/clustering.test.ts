import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ClusterPoint, ClusterSummary } from '../types/data';

const loaders = vi.hoisted(() => ({ csv: vi.fn(), json: vi.fn() }));
vi.mock('d3', () => loaders);

const referenceRows = [
  { image_name: 'a.png', cluster_label: '0', pca_x: '2', pca_y: '-3', distance_to_centroid: '.1' },
  { image_name: 'b.png', cluster_label: '1', pca_x: '-4', pca_y: '7', distance_to_centroid: '.2' },
  { image_name: 'c.png', cluster_label: '1', pca_x: '9', pca_y: '5', distance_to_centroid: '.3' },
];
const hierarchyRows = referenceRows.map((row, index) => ({
  ...row, cluster_label: String(index % 2), pca_y: String(-Number(row.pca_y)), distance_to_centroid: String(.5 + index / 10),
}));
const hierarchySummary: ClusterSummary = {
  sourceFile: 'model_eba.csv', filePath: '/hierarchical/model_eba_clusters.csv', model: 'model', roi: 'eba',
  clusteringMethod: 'hierarchical', distanceMetric: 'correlation', nImages: 3, bestK: 2, silhouette: .2,
  clusterSizes: { '0': 2, '1': 1 }, dendrogramPng: '/hierarchical/model_eba_dendrogram.png',
};

type RawRow = Record<string, string>;
function parseRows(rows: RawRow[], parser: (row: RawRow, index: number) => ClusterPoint | null) {
  return rows.map((row, index) => parser(row, index));
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
});
afterEach(() => vi.unstubAllGlobals());

describe('hierarchical reference view loading', () => {
  it('loads fixed reference geometry and sign alignment without changing reference labels or distances', async () => {
    loaders.csv.mockImplementation(async (path: string, parser) => parseRows(
      path.startsWith('/hierarchical/') ? hierarchyRows : referenceRows,
      parser,
    ));
    const { loadVoxelClusterView } = await import('./clustering');
    const loaded = await loadVoxelClusterView([hierarchySummary], { model: 'model', roi: 'eba' });

    expect(loaded?.uploadAxisAlignment).toEqual({ xSign: 1, ySign: -1 });
    expect(loaded?.points.map((point) => point.clusterLabel)).toEqual(['0', '1', '0']);
    expect(loaded?.points.map((point) => point.pcaY)).toEqual([3, -7, -5]);
    expect(loaded?.points.map((point) => point.distanceToCentroid)).toEqual([.5, .6, .7]);
    expect(loaded?.groups.map((group) => [group.label, group.size])).toEqual([['0', 2], ['1', 1]]);
    expect(loaded?.summary?.dendrogramPng).toBe(hierarchySummary.dendrogramPng);
    expect(loaders.csv).toHaveBeenCalledTimes(2);
  });

  it('keeps the hierarchy chart available if the alignment source fails and retries failed sources', async () => {
    loaders.csv.mockImplementation(async (path: string, parser) => {
      if (!path.startsWith('/hierarchical/')) throw new Error('Reference alignment source unavailable');
      return parseRows(hierarchyRows, parser);
    });
    const { loadVoxelClusterView } = await import('./clustering');
    const selected = { model: 'model', roi: 'eba' };
    const loaded = await loadVoxelClusterView([hierarchySummary], selected);
    expect(loaded?.points).toHaveLength(3);
    expect(loaded?.uploadAxisAlignment).toBeNull();

    loaders.csv.mockImplementation(async (path: string, parser) => parseRows(
      path.startsWith('/hierarchical/') ? hierarchyRows : referenceRows,
      parser,
    ));
    const retry = await loadVoxelClusterView([hierarchySummary], selected);
    expect(retry?.uploadAxisAlignment).toEqual({ xSign: 1, ySign: -1 });
    expect(loaders.csv).toHaveBeenCalledTimes(3);
  });

  it('fails closed on changed source geometry, preserving the reference chart', async () => {
    loaders.csv.mockImplementation(async (path: string, parser) => parseRows(
      path.startsWith('/hierarchical/') ? hierarchyRows : referenceRows.map((row) => ({ ...row, pca_x: String(Number(row.pca_x) + 2) })),
      parser,
    ));
    const { loadVoxelClusterView } = await import('./clustering');
    const loaded = await loadVoxelClusterView([hierarchySummary], { model: 'model', roi: 'eba' });
    expect(loaded?.uploadAxisAlignment).toBeNull();
    expect(loaded?.points).toHaveLength(3);
  });

  it('reuses the k-means reference load and leaves the existing k-means view unchanged', async () => {
    loaders.csv.mockImplementation(async (path: string, parser) => parseRows(
      path.startsWith('/hierarchical/') ? hierarchyRows : referenceRows,
      parser,
    ));
    const { loadVoxelClusterView } = await import('./clustering');
    const selected = { model: 'model', roi: 'eba' };
    const kmeansSummary = { ...hierarchySummary, filePath: undefined, clusteringMethod: undefined, distanceMetric: undefined };
    const kmeans = await loadVoxelClusterView([kmeansSummary], selected);
    await loadVoxelClusterView([hierarchySummary], selected);
    await loadVoxelClusterView([hierarchySummary], selected);
    expect(loaders.csv).toHaveBeenCalledTimes(2);
    expect(kmeans?.uploadAxisAlignment).toBeUndefined();
    expect(kmeans?.points.map((point) => point.pcaY)).toEqual([-3, 7, 5]);
  });

  it('continues to reject a missing hierarchy chart rather than disguising it as an alignment failure', async () => {
    loaders.csv.mockImplementation(async (path: string, parser) => {
      if (path.startsWith('/hierarchical/')) throw new Error('Hierarchy CSV missing');
      return parseRows(referenceRows, parser);
    });
    const { loadVoxelClusterView } = await import('./clustering');
    await expect(loadVoxelClusterView([hierarchySummary], { model: 'model', roi: 'eba' })).rejects.toThrow('Hierarchy CSV missing');
  });
});

describe('visual feature alignment loading', () => {
  it('aligns the actual loaded visual PCA displays without fetching the reference source again', async () => {
    loaders.json.mockImplementation(async (path: string) => {
      if (path.endsWith('/voxel_cluster_summary.json')) return [];
      if (path.endsWith('/visual_cluster_summary.json')) return { n_images: 3, best_k: 2, cluster_sizes: {} };
      throw new Error('Optional data not present in fixture');
    });
    loaders.csv.mockImplementation(async (path: string, parser) => {
      if (!path.endsWith('/visual_clusters.csv')) throw new Error('Optional data not present in fixture');
      const sign = path.includes('/visual_clusters_hierarchical_corr/') ? -1 : 1;
      return parseRows(referenceRows.map((row) => ({
        ...row, pca_x: String(Number(row.pca_x) * sign), plot_x: String(Number(row.pca_x) * sign), plot_y: row.pca_y,
      })), parser);
    });
    vi.stubGlobal('fetch', vi.fn(async () => ({ text: async () => '' })));
    const { loadClusteringData } = await import('./clustering');
    const data = await loadClusteringData();
    expect(data.visualViewByMethod.hierarchical_correlation?.uploadAxisAlignment).toEqual({ xSign: -1, ySign: 1 });
    expect(data.visualView?.uploadAxisAlignment).toBeUndefined();
    expect(loaders.csv.mock.calls.filter(([path]) => path.endsWith('/visual_clusters.csv'))).toHaveLength(2);
  });
});
