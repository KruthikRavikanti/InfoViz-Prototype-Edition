import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { ClusterPoint, ClusterView } from '../types/data';
import type { SharedResponsePoint } from '../utils/responseComparison';
import { ClusterScatter } from './ClusteringPanel';
import { pinnedClusterRelationship } from './LinkedComparisonClusters';

const pin = (name: string): SharedResponsePoint => ({
  imageId: name.replace('.png', ''), imageName: name, imageUrl: `/images/${name}`,
  meanA: 0, meanB: 0, rankA: 1, rankB: 1, zA: 0, zB: 0, gap: 0, gapDirection: null,
});
const pins = [pin('image_101.png'), pin('image_026.png')];
function view(labels: string[]): ClusterView {
  return {
    summary: null, groups: [], coordinateMode: 'pca',
    points: pins.map((p, i): ClusterPoint => ({
      imageName: p.imageName, imageUrl: p.imageUrl, clusterLabel: labels[i], value: 0, rank: i + 1,
      pcaX: i, pcaY: i, plotX: null, plotY: null, tsneX: null, tsneY: null, distanceToCentroid: 0,
    })),
  };
}

describe('linked pinned image clusters', () => {
  it('compares pair membership within each model, irrespective of cross-model cluster numbers', () => {
    expect(pinnedClusterRelationship(view(['0', '1']), pins)).toBe('separate');
    expect(pinnedClusterRelationship(view(['7', '7']), pins)).toBe('same');
  });

  it('does not claim a relationship for missing or incomplete pinned data', () => {
    expect(pinnedClusterRelationship(view(['0', '1']), pins.slice(0, 1))).toBe('incomplete');
    expect(pinnedClusterRelationship(view(['0', '1']), [pins[0], pin('absent.png')])).toBe('incomplete');
  });

  it('draws both neutral pin shapes without filtering or recoloring the reference points', () => {
    const markup = renderToStaticMarkup(<ClusterScatter
      view={view(['0', '1'])} onOpenImage={() => {}} searchQuery="" categoryHighlight={null}
      imageCategories={new Map()} pinnedImages={pins.map((p, index) => ({ imageName: p.imageName, index }))}
    />);
    expect((markup.match(/class="cluster-reference-point"/g) ?? []).length).toBe(2);
    expect(markup).toContain('fill="#4F5FA6"');
    expect(markup).toContain('fill="#C0392B"');
    expect(markup).toContain('data-pin-shape="circle"');
    expect(markup).toContain('data-pin-shape="diamond"');
    expect(markup).toContain('data-pinned-image="image_101.png" data-pinned-cluster="0"');
    expect(markup).toContain('data-pinned-image="image_026.png" data-pinned-cluster="1"');
    expect(markup).toContain('1 · image_101');
    expect(markup).toContain('2 · image_026');
  });
});
