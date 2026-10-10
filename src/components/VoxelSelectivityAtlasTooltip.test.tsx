import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { SelectivityAtlasHoverDatum, SelectivityBrainHoverDatum } from '../types/selectivityAtlas';
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
  it('adds native source coordinates for cortical markers without relabeling them as activation', () => {
    const hovered: SelectivityBrainHoverDatum = {
      voxelIndex: 180, roi: 'eba', roiVoxelIndex: 24, clusterLabel: 3,
      umapX: 2, umapY: -3, umapZ: 5, distanceToCentroid: 0.123456,
      canvasX: 200, canvasY: 100, hemisphere: 'rh', vertexIndex: 31403,
      volumeCoordinateZyx: [64, 30, 45], mappingWeight: 0.4423, sourceCount: 2,
    };
    const markup = renderToStaticMarkup(<VoxelSelectivityAtlasTooltip hovered={hovered} />);
    expect(markup).toContain('EBA voxel 24');
    expect(markup).toContain('Cluster C3');
    expect(markup).toContain('PCA centroid distance 0.123');
    expect(markup).toContain('RH · source voxel [z, y, x]: [64, 30, 45]');
    expect(markup).toContain('Supported surface vertex: 31,403');
    expect(markup).toContain('this marker identifies one source voxel');
    expect(markup).not.toContain('activation');
    expect(markup).not.toContain('UMAP');
    const hiddenMarkup = renderToStaticMarkup(<VoxelSelectivityAtlasTooltip hovered={{ ...hovered, occluded: true }} />);
    expect(hiddenMarkup).toContain('Hidden by cortex · rotate to inspect.');
    expect(hiddenMarkup).toContain('data-voxel-index="180"');
  });
});
