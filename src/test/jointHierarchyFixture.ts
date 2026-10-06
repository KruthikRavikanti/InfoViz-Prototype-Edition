import type { JointHierarchy } from '../types/session';

/** Small, complete joint tree for component and overlay tests. */
export function jointHierarchyFixture(
  referenceIds = ['reference-a.png', 'reference-b.png'],
  stimulusId = 'stimulus-a',
): JointHierarchy {
  return {
    schemaVersion: 1,
    method: 'hierarchical_correlation',
    distanceMetric: 'correlation',
    linkageMethod: 'average',
    normalization: 'reference_zscore',
    referenceCount: 2,
    uploadedCount: 1,
    clusterCount: 2,
    silhouette: .3,
    members: [
      { kind: 'reference', id: referenceIds[0], clusterId: '0', distanceToMedoid: 0 },
      { kind: 'reference', id: referenceIds[1], clusterId: '1', distanceToMedoid: 0 },
      { kind: 'uploaded', id: stimulusId, clusterId: '1', distanceToMedoid: .23 },
    ],
    linkage: [[1, 2, .23, 2], [0, 3, 1.1, 3]],
    medoidIndices: [0, 1],
  };
}
