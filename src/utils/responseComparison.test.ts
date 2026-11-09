import { describe, expect, it } from 'vitest';
import { csvParse } from 'd3';
import referenceCsv from '../../public/data/combined_image_means.csv?raw';
import type { EvidenceImage } from '../types/data';
import { rankImagesByColumn } from './evidence';
import { buildSharedResponseComparison, responseAxisDomain } from './responseComparison';

function image(id: number, value: number, rank = id, origin?: 'reference' | 'uploaded'): EvidenceImage {
  const imageName = `image_${String(id).padStart(3, '0')}.png`;
  return { imageName, imageUrl: `/images/${imageName}`, value, rank, origin };
}

describe('shared standardized predicted responses', () => {
  it('joins by image ID, uses only the shared population, and preserves existing ranks', () => {
    const result = buildSharedResponseComparison(
      [image(9, 100), image(1, 0, 19), image(2, 2, 12), image(3, 4, 4)],
      [image(3, 0, 27), image(8, 1000), image(1, 4, 8), image(2, 2, 14)],
    );
    expect(result.unavailableReason).toBeNull();
    expect(result.points.map(point => point.imageId)).toEqual(['image_001', 'image_002', 'image_003']);
    expect(result.meanA).toBe(2);
    expect(result.meanB).toBe(2);
    expect(result.sdA).toBeCloseTo(Math.sqrt(8 / 3), 14);
    expect(result.sdB).toBeCloseTo(Math.sqrt(8 / 3), 14);
    expect(result.points[0]).toMatchObject({ rankA: 19, rankB: 8 });
    expect(result.points[0].zA).toBeCloseTo(-Math.sqrt(1.5), 14);
    expect(result.points[0].zB).toBeCloseTo(Math.sqrt(1.5), 14);
  });

  it('excludes uploads and non-finite predictions without matching them to references', () => {
    const result = buildSharedResponseComparison(
      [image(1, 0), image(2, 1), image(3, NaN), image(1, 100, 1, 'uploaded')],
      [image(1, 1), image(2, 0), image(3, 3)],
    );
    expect(result.unavailableReason).toBeNull();
    expect(result.points).toHaveLength(2);
    expect(result.meanA).toBe(.5);
    expect(result.meanB).toBe(.5);
  });

  it('fails closed for duplicate IDs, insufficient matches, or constant responses', () => {
    expect(buildSharedResponseComparison([image(1, 0), image(1, 1)], [image(1, 2), image(2, 3)]).unavailableReason)
      .toContain('duplicate');
    expect(buildSharedResponseComparison([image(1, 0)], [image(1, 2)]).points).toEqual([]);
    const constant = buildSharedResponseComparison([image(1, 1), image(2, 1)], [image(1, 2), image(2, 3)]);
    expect(constant.points).toEqual([]);
    expect(constant.sdA).toBe(0);
    expect(constant.unavailableReason).toContain('variation');
  });

  it('handles very small nonzero variation and leaves identical model responses unhighlighted', () => {
    const images = [image(1, 1e-200), image(2, 2e-200), image(3, 3e-200)];
    const result = buildSharedResponseComparison(images, images);
    expect(result.unavailableReason).toBeNull();
    expect(result.sdA).toBeGreaterThan(0);
    expect(result.points.every(point => point.gap === 0 && point.gapDirection === null)).toBe(true);
  });

  it('selects two nonzero extrema per direction and breaks equal-gap ties by filename', () => {
    const a = [image(3, 0), image(2, 0), image(1, 0), image(6, 1), image(5, 1), image(4, 1)];
    const b = a.map(point => ({ ...point, value: 1 - point.value }));
    const result = buildSharedResponseComparison(a, b);
    expect(result.points.filter(point => point.gapDirection === 'b').map(point => point.imageId))
      .toEqual(['image_001', 'image_002']);
    expect(result.points.filter(point => point.gapDirection === 'a').map(point => point.imageId))
      .toEqual(['image_004', 'image_005']);
    const domain = responseAxisDomain(result.points);
    expect(domain[0]).toBeLessThan(-1);
    expect(domain[1]).toBeGreaterThan(1);
    expect(responseAxisDomain([])).toEqual([-1, 1]);
  });

  it('matches the independently audited 185-image RN50/ViT-B32 PPA predictions', () => {
    const rows = csvParse(referenceCsv, row => ({ ...row, image_name: String(row.image_name ?? '') }));
    const result = buildSharedResponseComparison(
      rankImagesByColumn(rows, 'clip_rn50_ppa'), rankImagesByColumn(rows, 'clip_vit_b32_ppa'),
    );
    expect(result.unavailableReason).toBeNull();
    expect(result.points).toHaveLength(185);
    expect(result.meanA).toBeCloseTo(-.2432765681196501, 14);
    expect(result.meanB).toBeCloseTo(-.2662515031030083, 14);
    expect(result.sdA).toBeCloseTo(.4921282716270678, 14);
    expect(result.sdB).toBeCloseTo(.5455536348452827, 14);
    const ankle = result.points.find(point => point.imageId === 'image_101')!;
    const infant = result.points.find(point => point.imageId === 'image_026')!;
    expect(ankle).toMatchObject({ rankA: 105, rankB: 175, gapDirection: 'a' });
    expect(ankle.zA).toBeCloseTo(-.343581337149067, 13);
    expect(ankle.zB).toBeCloseTo(-1.311291405696829, 13);
    expect(ankle.gap).toBeCloseTo(-.967710068547762, 13);
    expect(infant).toMatchObject({ rankA: 167, rankB: 116, gapDirection: null });
    expect(infant.gap).toBeCloseTo(.7668051754729253, 13);
    expect(result.points.filter(point => point.gapDirection === 'b').map(point => point.imageId))
      .toEqual(['image_027', 'image_127']);
    expect(result.points.filter(point => point.gapDirection === 'a').map(point => point.imageId))
      .toEqual(['image_099', 'image_101']);
    const domain = responseAxisDomain(result.points);
    expect(result.points.every(point => point.zA > domain[0] && point.zA < domain[1]
      && point.zB > domain[0] && point.zB < domain[1])).toBe(true);
  });
});
