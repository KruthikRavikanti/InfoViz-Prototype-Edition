import { describe, expect, it } from 'vitest';
import type { EvidenceImage, JsonScoreObject } from '../types/data';
import { buildAggregateHeatmapCells } from './heatmap';
import { mergeReferenceAndSessionRanks } from './sessionEvidence';

function reference(rank: number): EvidenceImage {
  return {
    imageName: `image_${rank}.png`,
    imageUrl: `/images/image_${rank}.png`,
    value: 10 - rank,
    rank,
    origin: 'reference',
  };
}

function uploaded(name: string, rank: number): EvidenceImage {
  return {
    imageName: name,
    imageUrl: `http://core/${name}`,
    value: 20 - rank,
    rank,
    origin: 'uploaded',
    stimulusId: name,
    sessionRank: rank,
    sessionCount: 5,
  };
}

describe('session ranking merge', () => {
  it('uses server session ranks while preserving reference order', () => {
    const merged = mergeReferenceAndSessionRanks(
      [reference(1), reference(2), reference(3)],
      [uploaded('first-upload', 2), uploaded('second-upload', 5)],
      5,
    );

    expect(merged.map((image) => image.imageName)).toEqual([
      'image_1.png',
      'first-upload',
      'image_2.png',
      'image_3.png',
      'second-upload',
    ]);
    expect(merged.map((image) => image.rank)).toEqual([1, 2, 3, 4, 5]);
    expect(merged.filter((image) => image.origin === 'reference').map((image) => image.referenceRank))
      .toEqual([1, 2, 3]);
  });

  it('does not mutate model performance scores or ranks', () => {
    const scores: JsonScoreObject = {
      ffa: { clip_rn50: 0.8, alexnet: 0.4 },
      ppa: { clip_rn50: 0.7, alexnet: 0.5 },
      eba: { clip_rn50: 0.6, alexnet: 0.3 },
      Overall: { clip_rn50: 0.7, alexnet: 0.4 },
    };
    const before = buildAggregateHeatmapCells(scores, ['ffa', 'ppa', 'eba'], ['clip_rn50', 'alexnet']);
    mergeReferenceAndSessionRanks([reference(1), reference(2), reference(3)], [uploaded('upload', 2)], 4);
    const after = buildAggregateHeatmapCells(scores, ['ffa', 'ppa', 'eba'], ['clip_rn50', 'alexnet']);

    expect(after).toEqual(before);
  });
});
