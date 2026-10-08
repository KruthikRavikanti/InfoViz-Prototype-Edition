import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { SelectivityAtlasHoverDatum } from '../types/selectivityAtlas';
import { VoxelSelectivityAtlasTooltip } from './VoxelSelectivityAtlasTooltip';

describe('voxel selectivity tooltip', () => {
  it('preserves voxel identity, cluster and PCA distance for mouse and keyboard inspection', () => {
    const hovered: SelectivityAtlasHoverDatum = {
      voxelIndex: 180, roi: 'eba', roiVoxelIndex: 24, clusterLabel: 3,
      umapX: 2, umapY: -3, umapZ: 5, distanceToCentroid: 0.123456,
      canvasX: 900, canvasY: 600,
    };
    const markup = renderToStaticMarkup(<VoxelSelectivityAtlasTooltip hovered={hovered} />);
    expect(markup).toContain('data-voxel-index="180"');
    expect(markup).toContain('role="status"');
    expect(markup).toContain('EBA voxel 24');
    expect(markup).toContain('Cluster C3');
    expect(markup).toContain('PCA centroid distance 0.123');
    expect(markup).not.toContain('left:900px'); // Placement is measured and clamped after mount.
    expect(markup).not.toContain('activation');
    expect(markup).not.toContain('Outside current view');
    const offscreenMarkup = renderToStaticMarkup(<VoxelSelectivityAtlasTooltip hovered={{ ...hovered, outsideView: true }} />);
    expect(offscreenMarkup).toContain('Outside current view · zoom out or reset.');
    expect(offscreenMarkup).toContain('data-voxel-index="180"');
  });
});
