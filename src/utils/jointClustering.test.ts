import { describe, expect, it } from 'vitest';
import type { ClusterPoint, ClusterView } from '../types/data';
import type { JointHierarchy } from '../types/session';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { buildJointClusterView, jointClusterMedoidName } from './jointClustering';

function fixtures() {
  const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
  const points: ClusterPoint[] = ['a.png', 'b.png', 'c.png'].map((imageName, index) => ({
    imageName, imageUrl: `/images/${imageName}`, clusterLabel: '9', value: 5, rank: index + 1,
    pcaX: index, pcaY: -index, plotX: index, plotY: -index, tsneX: null, tsneY: null,
    distanceToCentroid: .8,
  }));
  const view: ClusterView = {
    points, groups: [{ label: '9', size: 3, points, images: [] }], coordinateMode: 'plot',
    uploadAxisAlignment: { xSign: 1, ySign: -1 },
    summary: {
      nImages: 3, bestK: 1, silhouette: .3, clusterSizes: { '9': 3 },
      dendrogramPng: '/old.png', linkageCsv: '/old.csv', resultCsv: '/old-result.csv',
      filePath: '/old-clusters.csv', pcaDimUsed: 20, projectionUsedForClustering: 'PCA',
    },
  };
  const hierarchy: JointHierarchy = {
    schemaVersion: 1, method: 'hierarchical_correlation', distanceMetric: 'correlation',
    linkageMethod: 'average', normalization: 'reference_zscore',
    referenceCount: 3, uploadedCount: 1, clusterCount: 2, silhouette: .75,
    members: [
      { kind: 'reference', id: 'a.png', clusterId: '0', distanceToMedoid: 0 },
      { kind: 'reference', id: 'b.png', clusterId: '0', distanceToMedoid: .2 },
      { kind: 'reference', id: 'c.png', clusterId: '0', distanceToMedoid: .4 },
      { kind: 'uploaded', id: 'stimulus-a', clusterId: '1', distanceToMedoid: 0 },
    ],
    linkage: [[0, 1, .2, 2], [2, 4, .4, 3], [3, 5, 1.2, 4]],
    medoidIndices: [0, 3],
  };
  return { session, view, hierarchy };
}

describe('joint hierarchy reference-view adaptation', () => {
  it('uses joint labels, includes upload-only groups, and preserves every fixed coordinate', () => {
    const { session, view, hierarchy } = fixtures();
    const original = structuredClone({ session, view, hierarchy });
    const joint = buildJointClusterView(view, hierarchy, session)!;
    expect(joint.points).toHaveLength(3);
    expect(joint.points.map((point) => point.clusterLabel)).toEqual(['0', '0', '0']);
    expect(joint.points.map(({ pcaX, pcaY, plotX, plotY, tsneX, tsneY }) => ({ pcaX, pcaY, plotX, plotY, tsneX, tsneY })))
      .toEqual(view.points.map(({ pcaX, pcaY, plotX, plotY, tsneX, tsneY }) => ({ pcaX, pcaY, plotX, plotY, tsneX, tsneY })));
    expect(joint.groups.map(({ label, size, points }) => ({ label, size, count: points.length })))
      .toEqual([{ label: '0', size: 3, count: 3 }, { label: '1', size: 1, count: 0 }]);
    expect(joint.groups[1].images[0]).toMatchObject({
      origin: 'uploaded', stimulusId: 'stimulus-a', imageName: 'uploaded.png', value: 0,
    });
    expect(joint.summary).toMatchObject({
      nImages: 4, bestK: 2, silhouette: .75, clusterSizes: { '0': 3, '1': 1 },
      clusteringMethod: 'hierarchical', distanceMetric: 'correlation', linkageMethod: 'average',
      normalization: 'reference_zscore', projectionUsedForClustering: 'full standardized responses/features (not PCA)',
    });
    for (const field of ['dendrogramPng', 'linkageCsv', 'resultCsv', 'filePath', 'pcaDimUsed'] as const) {
      expect(joint.summary?.[field]).toBeUndefined();
    }
    expect(joint.uploadAxisAlignment).toEqual(view.uploadAxisAlignment);
    expect({ session, view, hierarchy }).toEqual(original);
  });

  it('matches reference IDs rather than relying on hierarchy member order', () => {
    const { session, view, hierarchy } = fixtures();
    [hierarchy.members[0], hierarchy.members[2]] = [hierarchy.members[2], hierarchy.members[0]];
    hierarchy.medoidIndices[0] = 2;
    const joint = buildJointClusterView(view, hierarchy, session)!;
    expect(joint.points.map((point) => point.distanceToCentroid)).toEqual([0, .2, .4]);
    expect(joint.groups[0].images[0].imageName).toBe('a.png');
  });

  it('accepts upload medoids without treating the old assignment as the new cluster', () => {
    const { session, view, hierarchy } = fixtures();
    hierarchy.members[0].clusterId = '1';
    hierarchy.members[0].distanceToMedoid = .15;
    hierarchy.members[1].distanceToMedoid = 0;
    hierarchy.medoidIndices = [1, 3];
    const joint = buildJointClusterView(view, hierarchy, session)!;
    expect(joint.groups[1].size).toBe(2);
    expect(joint.groups[1].images[0].origin).toBe('uploaded');
    expect(jointClusterMedoidName(hierarchy, '1', session)).toBe('uploaded.png');
    expect(jointClusterMedoidName(hierarchy, '0', session)).toBe('b.png');
    expect(session.roiResults.ffa.stimuli[0].hierarchicalAssignment.clusterId).toBe('2');
    expect(joint.points[0].clusterLabel).toBe('1');
  });

  it('keeps joint relabeling available when an upload projection is not aligned', () => {
    const { session, view, hierarchy } = fixtures();
    view.uploadAxisAlignment = null;
    const joint = buildJointClusterView(view, hierarchy, session)!;
    expect(joint.summary?.nImages).toBe(4);
    expect(joint.uploadAxisAlignment).toBeNull();
  });

  it.each(['missing-reference', 'duplicate-reference', 'unknown-upload', 'duplicate-upload', 'wrong-count', 'bad-medoid'])
  ('fails closed for incompatible member sets: %s', (caseName) => {
    const { session, view, hierarchy } = fixtures();
    if (caseName === 'missing-reference') hierarchy.members[0].id = 'absent.png';
    if (caseName === 'duplicate-reference') hierarchy.members[1].id = 'a.png';
    if (caseName === 'unknown-upload') hierarchy.members[3].id = 'unknown-upload';
    if (caseName === 'duplicate-upload') {
      hierarchy.members[2] = { ...hierarchy.members[3] };
    }
    if (caseName === 'wrong-count') hierarchy.referenceCount = 185;
    if (caseName === 'bad-medoid') hierarchy.medoidIndices[1] = 0;
    expect(buildJointClusterView(view, hierarchy, session)).toBeNull();
  });

  it('fails closed when the joint result is absent', () => {
    const { session, view, hierarchy } = fixtures();
    expect(buildJointClusterView(view, undefined, session)).toBeNull();
    expect(buildJointClusterView(null, hierarchy, session)).toBeNull();
    expect(jointClusterMedoidName(hierarchy, 'no-cluster', session)).toBeUndefined();
  });

  it('does not relabel a different model’s reference predictions with this session', () => {
    const { session, view, hierarchy } = fixtures();
    view.summary!.model = 'alexnet';
    expect(buildJointClusterView(view, hierarchy, session)).toBeNull();
  });
});
