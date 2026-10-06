import { describe, expect, it } from 'vitest';
import { CANONICAL_ROIS, SUBJECT_ONE_ROI_VOXEL_COUNTS, type CanonicalRoi } from '../types/session';
import type { BrainMapImage, BrainMapImageSource } from './brainMapSelection';
import { buildBrainVoxelImagePeaks, getBrainVoxelImagePeak } from './brainVoxelImagePeaks';

function image(id: string, value: number, source: BrainMapImageSource = 'uploaded'): BrainMapImage {
  return {
    id,
    source,
    displayName: `${id}.png`,
    imageUrl: `/images/${id}.png`,
    result: {
      stimulusId: id,
      roiResponses: Object.fromEntries(CANONICAL_ROIS.map((roi) => [
        roi, Array.from({ length: SUBJECT_ONE_ROI_VOXEL_COUNTS[roi] }, () => value),
      ])) as Record<CanonicalRoi, number[]>,
    },
  };
}

describe('brain-map source-voxel peak images', () => {
  it('compares signed predictions and selects the least-negative response when all images are negative', () => {
    const stronglyNegative = image('negative-large-magnitude', -9);
    const lessNegative = image('negative-highest', -0.2);
    const peak = getBrainVoxelImagePeak(buildBrainVoxelImagePeaks([stronglyNegative, lessNegative], 'uploaded'), 'ffa', 0);
    expect(peak?.image).toBe(lessNegative);
    expect(peak?.response).toBe(-0.2);
    expect(peak?.tieCount).toBe(1);
  });

  it('indexes every canonical ROI and response column separately without averaging or normalizing', () => {
    const a = image('a', 0);
    const b = image('b', 1);
    a.result.roiResponses.ffa[155] = 4.25;
    a.result.roiResponses.ppa[0] = 17;
    a.result.roiResponses.eba[517] = 1.01;
    const index = buildBrainVoxelImagePeaks([a, b], 'uploaded');
    expect(index.status).toBe('ready');
    for (const roi of CANONICAL_ROIS) expect(index.peaks[roi]).toHaveLength(SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]);
    expect(getBrainVoxelImagePeak(index, 'ffa', 0)?.image).toBe(b);
    expect(getBrainVoxelImagePeak(index, 'ffa', 155)?.response).toBe(4.25);
    expect(getBrainVoxelImagePeak(index, 'ppa', 0)?.image).toBe(a);
    expect(getBrainVoxelImagePeak(index, 'ppa', 1)?.image).toBe(b);
    expect(getBrainVoxelImagePeak(index, 'eba', 517)?.response).toBe(1.01);
  });

  it('counts exact ties and selects the first supplied image deterministically', () => {
    const a = image('a', 2);
    const b = image('b', 2);
    const smaller = image('smaller', 1);
    const higher = image('higher', 3);
    higher.result.roiResponses.ffa[0] = 2;
    const index = buildBrainVoxelImagePeaks([a, smaller, b, higher], 'uploaded');
    expect(getBrainVoxelImagePeak(index, 'ffa', 0)).toEqual({ image: a, response: 2, tieCount: 3 });
    expect(getBrainVoxelImagePeak(index, 'ffa', 1)).toEqual({ image: higher, response: 3, tieCount: 1 });
    expect(getBrainVoxelImagePeak(buildBrainVoxelImagePeaks([b, a], 'uploaded'), 'ffa', 0)?.image).toBe(b);
  });

  it('scopes candidates strictly to uploads or reference images even when their IDs collide', () => {
    const upload = image('same-id', 0.2, 'uploaded');
    const reference = image('same-id', 9, 'reference');
    const uploadIndex = buildBrainVoxelImagePeaks([reference, upload], 'uploaded');
    const referenceIndex = buildBrainVoxelImagePeaks([upload, reference], 'reference');
    expect(uploadIndex.source).toBe('uploaded');
    expect(uploadIndex.candidateCount).toBe(1);
    expect(getBrainVoxelImagePeak(uploadIndex, 'ppa', 3)?.image).toBe(upload);
    expect(getBrainVoxelImagePeak(referenceIndex, 'ppa', 3)?.image).toBe(reference);
  });

  it('does not invalidate one collection because the other contains malformed predictions', () => {
    const upload = image('valid-upload', 0.2);
    const reference = image('invalid-reference', 99, 'reference');
    reference.result.roiResponses.eba = [];
    expect(buildBrainVoxelImagePeaks([reference, upload], 'uploaded').status).toBe('ready');
    expect(buildBrainVoxelImagePeaks([reference, upload], 'reference').status).toBe('invalid');
  });

  it('represents empty or unavailable collections explicitly', () => {
    for (const index of [
      buildBrainVoxelImagePeaks([], 'reference'),
      buildBrainVoxelImagePeaks([image('upload', 1)], 'reference'),
    ]) {
      expect(index.status).toBe('empty');
      expect(index.candidateCount).toBe(0);
      expect(index.peaks).toEqual({ ffa: [], ppa: [], eba: [] });
      expect(getBrainVoxelImagePeak(index, 'ffa', 0)).toBeNull();
    }
  });

  it.each([NaN, Infinity, -Infinity, '4', undefined])('fails closed for a non-finite response: %s', (value) => {
    const valid = image('valid', 5);
    const invalid = image('invalid', 1);
    invalid.result.roiResponses.ppa[10] = value as number;
    const index = buildBrainVoxelImagePeaks([valid, invalid], 'uploaded');
    expect(index.status).toBe('invalid');
    expect(index.candidateCount).toBe(2);
    expect(getBrainVoxelImagePeak(index, 'ffa', 0)).toBeNull();
  });

  it('fails closed for missing vectors, sparse responses and incorrect Subject-1 dimensions', () => {
    const malformedImages = [image('short', 1), image('long', 1), image('missing-roi', 1), image('missing-result', 1), image('sparse', 1)];
    malformedImages[0].result.roiResponses.ffa.pop();
    malformedImages[1].result.roiResponses.eba.push(1);
    delete (malformedImages[2].result.roiResponses as Partial<Record<CanonicalRoi, number[]>>).ppa;
    delete (malformedImages[3] as Partial<BrainMapImage>).result;
    delete malformedImages[4].result.roiResponses.ffa[10];
    for (const malformed of malformedImages) {
      const index = buildBrainVoxelImagePeaks([image('valid', 2), malformed], 'uploaded');
      expect(index.status).toBe('invalid');
      expect(index.peaks).toEqual({ ffa: [], ppa: [], eba: [] });
    }
  });

  it('rejects query indices outside the original ROI response columns', () => {
    const index = buildBrainVoxelImagePeaks([image('valid', 1)], 'uploaded');
    for (const responseIndex of [-1, 0.5, NaN, Infinity, 156]) {
      expect(getBrainVoxelImagePeak(index, 'ffa', responseIndex)).toBeNull();
    }
    expect(getBrainVoxelImagePeak(index, 'not-a-roi' as CanonicalRoi, 0)).toBeNull();
    expect(getBrainVoxelImagePeak(index, 'eba', 517)?.response).toBe(1);
  });

  it('does not mutate or freeze inputs and protects its derived lookup metadata', () => {
    const a = image('a', 1);
    const b = image('b', 2);
    const inputs = Object.freeze([a, b]);
    const before = JSON.stringify(inputs);
    for (const input of inputs) {
      for (const roi of CANONICAL_ROIS) Object.freeze(input.result.roiResponses[roi]);
      Object.freeze(input.result.roiResponses);
      Object.freeze(input.result);
    }
    const index = buildBrainVoxelImagePeaks(inputs, 'uploaded');
    expect(JSON.stringify(inputs)).toBe(before);
    expect(Object.isFrozen(a)).toBe(false);
    expect(Object.isFrozen(index)).toBe(true);
    expect(Object.isFrozen(index.peaks)).toBe(true);
    expect(Object.isFrozen(index.peaks.ffa)).toBe(true);
    expect(Object.isFrozen(index.peaks.ffa[0])).toBe(true);
    expect(index.peaks.ffa[0].image).toBe(b);
  });
});
