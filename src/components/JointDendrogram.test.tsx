import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import type { ClusterPoint, ClusterView } from '../types/data';
import type { JointHierarchy } from '../types/session';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { buildJointClusterView } from '../utils/jointClustering';
import { buildJointDendrogramLayout, JointDendrogram } from './JointDendrogram';

function fixtures() {
  const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
  const points: ClusterPoint[] = ['a.png', 'b.png', 'c.png'].map((imageName, index) => ({
    imageName, imageUrl: `/images/${imageName}`, clusterLabel: '9', value: .9, rank: index + 1,
    pcaX: index, pcaY: index, plotX: index, plotY: index, tsneX: null, tsneY: null, distanceToCentroid: .9,
  }));
  const reference: ClusterView = { points, summary: null, groups: [], coordinateMode: 'pca' };
  const hierarchy: JointHierarchy = {
    schemaVersion: 1, method: 'hierarchical_correlation', distanceMetric: 'correlation',
    linkageMethod: 'average', normalization: 'reference_zscore',
    referenceCount: 3, uploadedCount: 1, clusterCount: 2, silhouette: .5,
    members: [
      { kind: 'reference', id: 'a.png', clusterId: '0', distanceToMedoid: 0 },
      { kind: 'reference', id: 'b.png', clusterId: '0', distanceToMedoid: .2 },
      { kind: 'reference', id: 'c.png', clusterId: '1', distanceToMedoid: .1 },
      { kind: 'uploaded', id: 'stimulus-a', clusterId: '1', distanceToMedoid: 0 },
    ],
    linkage: [[2, 3, .1, 2], [0, 1, .2, 2], [5, 4, 1.2, 4]], medoidIndices: [0, 3],
  };
  return { session, hierarchy, view: buildJointClusterView(reference, hierarchy, session)! };
}

describe('joint session dendrogram', () => {
  it('renders every reference and upload as a genuine interactive leaf', () => {
    const { session, view } = fixtures();
    const markup = renderToStaticMarkup(<JointDendrogram view={view} session={session} onOpenImage={vi.fn()} />);
    expect(markup.match(/data-joint-leaf="reference"/g)).toHaveLength(3);
    expect(markup.match(/data-joint-leaf="uploaded"/g)).toHaveLength(1);
    expect(markup.match(/data-linkage-node=/g)).toHaveLength(3);
    expect(markup.match(/role="button"/g)).toHaveLength(4);
    expect(markup).toContain('Uploaded stimulus uploaded.png, jointly clustered in cluster 1');
    expect(markup).toContain('data-link-key="uploaded:stimulus-a"');
    expect(markup).toContain('joint-dendrogram-upload-glow');
    expect(markup).toContain('Average-linkage correlation distance (1 − r)');
    expect(markup).toContain('data-merge-distance="1.2"');
    expect(markup).not.toContain('nearest-medoid assignment');
    expect(markup).not.toContain('<img');
  });

  it('uses the exact binary linkage and its true merge heights', () => {
    const { hierarchy } = fixtures();
    const layout = buildJointDendrogramLayout(hierarchy)!;
    expect(layout.leafOrder).toEqual([0, 1, 2, 3]);
    expect(layout.nodes.map((node) => node.distance)).toEqual([0, 0, 0, 0, .1, .2, 1.2]);
    expect(layout.nodes[4].clusterId).toBe('1');
    expect(layout.nodes[5].clusterId).toBe('0');
    expect(layout.nodes[6].clusterId).toBeNull();
    expect(layout.nodes[6].size).toBe(4);
  });

  it('can highlight the same uploaded leaf as the linked scatter charts', () => {
    const { session, view } = fixtures();
    const markup = renderToStaticMarkup(
      <JointDendrogram view={view} session={session} onOpenImage={vi.fn()} linkedItem={{ kind: 'uploaded', id: 'stimulus-a', label: 'uploaded.png' }} />,
    );
    expect(markup.match(/joint-dendrogram-leaf--highlighted/g)).toHaveLength(1);
  });

  it.each(['self-link', 'reused-child', 'invalid-count', 'unknown-child', 'missing-row', 'invalid-distance', 'nonmonotonic'])
  ('refuses a malformed linkage rather than drawing an invented tree: %s', (caseName) => {
    const { hierarchy } = fixtures();
    if (caseName === 'self-link') hierarchy.linkage[0][1] = 0;
    if (caseName === 'reused-child') hierarchy.linkage[1][0] = 2;
    if (caseName === 'invalid-count') hierarchy.linkage[0][3] = 3;
    if (caseName === 'unknown-child') hierarchy.linkage[0][0] = 4;
    if (caseName === 'missing-row') hierarchy.linkage.pop();
    if (caseName === 'invalid-distance') hierarchy.linkage[0][2] = Number.NaN;
    if (caseName === 'nonmonotonic') hierarchy.linkage[1][2] = .05;
    expect(buildJointDendrogramLayout(hierarchy)).toBeNull();
  });

  it('does not display a static reference dendrogram as a session dendrogram', () => {
    const { session, view } = fixtures();
    delete view.jointHierarchy;
    const markup = renderToStaticMarkup(<JointDendrogram view={view} session={session} onOpenImage={vi.fn()} />);
    expect(markup).toContain('The session dendrogram is unavailable.');
    expect(markup).not.toContain('<svg');
  });
});
