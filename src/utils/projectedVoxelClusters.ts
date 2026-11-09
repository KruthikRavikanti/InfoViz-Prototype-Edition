import type { BrainHemisphere, BrainMapAssets } from '../types/brainMap';
import type { SelectivityAtlasPoint } from '../types/selectivityAtlas';
import { CANONICAL_ROIS, type CanonicalRoi } from '../types/session';
import { validatePointIdentities } from './selectivityBrainProjection';

export type VoxelClusterContributor = { point: SelectivityAtlasPoint; weight: number };

export type ProjectedVoxelClusters = {
  /** Existing atlas labels; -1 means that no enabled voxel supports the vertex. */
  labels: Int16Array;
  support: Float64Array;
  clusterWeights: Map<number, Map<number, number>>;
  /** Source weights are aggregated if the mapper repeats a source/vertex edge. */
  contributors: Map<number, VoxelClusterContributor[]>;
};

/**
 * Categorical projection through the same Subject 1 sparse mapper as Brain Map.
 * Sum positive weights within each enabled ROI's existing voxel clusters, then
 * choose the strongest cluster. Exact ties go to the smaller cluster number.
 * No dilation, nearest-neighbor filling, or averaging of numeric labels occurs.
 */
export function projectVoxelClusters({ assets, points, enabledRois, hemisphere, vertexCount }: {
  assets: BrainMapAssets;
  points: SelectivityAtlasPoint[];
  enabledRois: ReadonlySet<CanonicalRoi>;
  hemisphere: BrainHemisphere;
  vertexCount: number;
}): ProjectedVoxelClusters {
  const expectedCount = assets.manifest.surfaces.inflated?.hemispheres[hemisphere].vertexCount;
  if (!Number.isInteger(vertexCount) || vertexCount < 0 || expectedCount !== vertexCount) {
    throw new Error('The cortical mesh does not match the mapper vertex ordering.');
  }
  const bySource = validatePointIdentities(assets, points);
  const labels = new Int16Array(vertexCount).fill(-1);
  const support = new Float64Array(vertexCount);
  const clusterWeights = new Map<number, Map<number, number>>();
  const sourcesByVertex = new Map<number, Map<string, VoxelClusterContributor>>();

  // Mapper order, rather than embedding/global atlas order, determines summation.
  for (const roi of CANONICAL_ROIS) {
    if (!enabledRois.has(roi)) continue;
    const mapping = assets.projection.rois[roi].mappings[hemisphere];
    if (mapping.responseIndices.length !== mapping.vertexIndices.length
      || mapping.responseIndices.length !== mapping.weights.length) {
      throw new Error('Cortical mapper arrays are incompatible.');
    }
    for (let edge = 0; edge < mapping.weights.length; edge += 1) {
      const sourceKey = `${roi}:${mapping.responseIndices[edge]}`;
      const point = bySource.get(sourceKey);
      const vertex = mapping.vertexIndices[edge];
      const weight = mapping.weights[edge];
      if (!point || !Number.isInteger(vertex) || vertex < 0 || vertex >= vertexCount
        || !Number.isFinite(weight) || weight < 0) {
        throw new Error('Cortical mapper contains an invalid source, weight, or surface vertex.');
      }
      if (weight === 0) continue;
      support[vertex] += weight;
      let weights = clusterWeights.get(vertex);
      if (!weights) { weights = new Map(); clusterWeights.set(vertex, weights); }
      weights.set(point.clusterLabel, (weights.get(point.clusterLabel) ?? 0) + weight);
      let sources = sourcesByVertex.get(vertex);
      if (!sources) { sources = new Map(); sourcesByVertex.set(vertex, sources); }
      const previous = sources.get(sourceKey);
      sources.set(sourceKey, { point, weight: (previous?.weight ?? 0) + weight });
    }
  }

  const contributors = new Map<number, VoxelClusterContributor[]>();
  for (const [vertex, weights] of clusterWeights) {
    let maximum = -Infinity;
    for (const [cluster, weight] of weights) {
      if (weight > maximum || (weight === maximum && cluster < labels[vertex])) {
        maximum = weight;
        labels[vertex] = cluster;
      }
    }
    // Stable order also resolves equal-weight source choices during inspection.
    contributors.set(vertex, [...sourcesByVertex.get(vertex)!.values()].sort((a, b) =>
      b.weight - a.weight
      || CANONICAL_ROIS.indexOf(a.point.roi) - CANONICAL_ROIS.indexOf(b.point.roi)
      || a.point.roiVoxelIndex - b.point.roiVoxelIndex));
  }
  return { labels, support, clusterWeights, contributors };
}
