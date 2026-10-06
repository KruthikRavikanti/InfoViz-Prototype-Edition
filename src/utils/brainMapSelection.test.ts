import { describe, expect, it } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { BRAIN_REFERENCE_VERSION } from '../services/brainReferenceAssets';
import { rawSessionFixture } from '../test/sessionFixture';
import type { BrainMapReferenceSet } from '../types/brainMap';
import {
  brainMapImageNeighbors,
  brainMapReferenceImages,
  brainMapUploadedImages,
  filterBrainMapImages,
} from './brainMapSelection';
import { fixedSessionResponseExtent } from './brainProjection';

function fixture() {
  const session = validateComparisonSession(rawSessionFixture(), 'http://core.example');
  session.comparison.referenceVersion = BRAIN_REFERENCE_VERSION;
  session.brainMapResults = {
    schemaVersion: 1,
    subjectId: 's1',
    responseSpace: 'nsd_1000_subject1_floc_t7_v1',
    roiVoxelCounts: { ffa: 156, ppa: 170, eba: 518 },
    stimuli: [{ stimulusId: 'stimulus-a', roiResponses: { ffa: [1], ppa: [-2], eba: [3] } }],
  };
  const references: BrainMapReferenceSet = {
    model: 'clip_rn50',
    referenceVersion: session.comparison.referenceVersion,
    responseSpace: 'nsd_1000_subject1_floc_t7_v1',
    stimuli: [
      { imageName: 'image_001.png', imageUrl: '/images/image_001.png', roiResponses: { ffa: [4], ppa: [-5], eba: [6] } },
      { imageName: 'image_002.png', imageUrl: '/images/image_002.png', roiResponses: { ffa: [7], ppa: [-8], eba: [9] } },
    ],
  };
  return { session, references };
}

describe('brain-map image collection selection', () => {
  it('keeps reference vectors raw and uses image identities independent of upload IDs', () => {
    const { references } = fixture();
    const images = brainMapReferenceImages(references);
    expect(images.map((image) => image.id)).toEqual(['image_001.png', 'image_002.png']);
    expect(images[0].source).toBe('reference');
    expect(images[0].result.stimulusId).toBe('reference:image_001.png');
    expect(images[0].result.roiResponses).toBe(references.stimuli[0].roiResponses);
  });

  it('only exposes upload vectors with a matching model and reference bundle', () => {
    const { session, references } = fixture();
    expect(brainMapUploadedImages(session, 'clip_rn50', references)).toHaveLength(1);
    expect(brainMapUploadedImages(session, 'alexnet', references)).toEqual([]);
    expect(brainMapUploadedImages(session, 'clip_rn50', { ...references, model: 'alexnet' })).toEqual([]);
    expect(brainMapUploadedImages(session, 'clip_rn50', { ...references, referenceVersion: 'different' })).toEqual([]);
    expect(brainMapUploadedImages(null, 'clip_rn50', references)).toEqual([]);
    session.brainMapResults = undefined;
    expect(brainMapUploadedImages(session, 'clip_rn50', references)).toEqual([]);
  });

  it('matches uploaded vectors by ID rather than assumed array order and leaves session data untouched', () => {
    const { session } = fixture();
    session.stimuli.push({ id: 'missing-vector', displayName: 'missing.png', imageUrl: '/missing.png', color: '#000000' });
    const before = JSON.stringify(session);
    const images = brainMapUploadedImages(session, 'clip_rn50');
    expect(images).toHaveLength(1);
    expect(images[0].result).toBe(session.brainMapResults!.stimuli[0]);
    expect(JSON.stringify(session)).toBe(before);
  });

  it('never colors uploads from an incompatible reference bundle before references load or after loading fails', () => {
    const { session } = fixture();
    expect(brainMapUploadedImages(session, 'clip_rn50')).toHaveLength(1);
    session.comparison.referenceVersion = '0'.repeat(64);
    expect(brainMapUploadedImages(session, 'clip_rn50')).toEqual([]);
  });

  it('rejects incompatible anatomical response metadata independently of reference loading', () => {
    const { session } = fixture();
    session.brainMapResults!.roiVoxelCounts.ffa = 155;
    expect(brainMapUploadedImages(session, 'clip_rn50')).toEqual([]);
    session.brainMapResults!.roiVoxelCounts.ffa = 156;
    const incompatible = session.brainMapResults as unknown as { subjectId: string; responseSpace: string };
    incompatible.subjectId = 's2';
    expect(brainMapUploadedImages(session, 'clip_rn50')).toEqual([]);
    incompatible.subjectId = 's1';
    incompatible.responseSpace = 'other-response-order';
    expect(brainMapUploadedImages(session, 'clip_rn50')).toEqual([]);
  });

  it('filters and navigates a bounded picker without changing the image collections', () => {
    const images = brainMapReferenceImages(fixture().references);
    expect(filterBrainMapImages(images, ' IMAGE_002 ')).toEqual([images[1]]);
    expect(filterBrainMapImages(images, '')).toBe(images);
    expect(filterBrainMapImages(images, 'absent')).toEqual([]);
    expect(brainMapImageNeighbors(images, images[0].id)).toEqual({ previous: undefined, next: images[1] });
    expect(brainMapImageNeighbors(images, images[1].id)).toEqual({ previous: images[0], next: undefined });
    expect(brainMapImageNeighbors(images, 'outside-filter')).toEqual({ previous: undefined, next: images[0] });
    expect(brainMapImageNeighbors([], 'outside-filter')).toEqual({ previous: undefined, next: undefined });
  });

  it('shares one symmetric response scale across every reference, upload, and ROI', () => {
    const { session, references } = fixture();
    const images = [...brainMapUploadedImages(session, 'clip_rn50', references), ...brainMapReferenceImages(references)];
    expect(fixedSessionResponseExtent(images.map((image) => image.result))).toBe(9);
  });
});
