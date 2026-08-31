import { describe, expect, it } from 'vitest';
import {
  CortexSessionError,
  assertSessionReferenceCompatible,
  extractSessionToken,
  validateComparisonSession,
  validateReferenceManifest,
} from './cortexSession';
import { rawReferenceManifestFixture, rawSessionFixture } from '../test/sessionFixture';

describe('Cortex comparison session validation', () => {
  it('parses a valid canonical session and keeps reference images same-origin', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');

    expect(session.comparison.dataset).toBe('nsd_1000');
    expect(session.comparison.rois).toEqual(['ffa', 'ppa', 'eba']);
    expect(session.roiResults.ffa.stimuli[0].referenceRank).toBe(5.5);
    expect(session.stimuli[0].imageUrl).toBe(
      'http://core.example:5556/comparison-sessions/opaque-token/images/stimulus-a',
    );
    expect(session.roiResults.ffa.stimuli[0].predecessor?.imageUrl).toBe('/images/image_005.png');
  });

  it('rejects incomplete ROI data', () => {
    const raw = rawSessionFixture();
    delete (raw.roi_results as Record<string, unknown>).ppa;
    expect(() => validateComparisonSession(raw, 'http://core.example:5556')).toThrow(CortexSessionError);
  });

  it('accepts fractional reference-neighbor midranks', () => {
    const raw = rawSessionFixture();
    raw.roi_results.ffa.stimuli[0].predecessor.rank = 4.5;
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.roiResults.ffa.stimuli[0].predecessor?.rank).toBe(4.5);
  });

  it('accepts upload reference rank 186 when the prediction is below all references', () => {
    const raw = rawSessionFixture();
    raw.roi_results.ffa.stimuli[0].reference_rank = 186;
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.roiResults.ffa.stimuli[0].referenceRank).toBe(186);
  });

  it('accepts visual neighbors without response values or ranks', () => {
    const raw = rawSessionFixture() as ReturnType<typeof rawSessionFixture> & { visual_results?: unknown };
    raw.visual_results = {
      stimuli: [{
        stimulus_id: 'stimulus-a',
        x: 1.2,
        y: -0.4,
        cluster_id: 1,
        centroid_distance: 0.31,
        nearest_reference_members: [{
          image_name: 'image_012.png',
          image_url: '/images/image_012.png',
          distance: 0.08,
        }],
        hierarchical_assignment: {
          method: 'nearest-medoid assignment',
          cluster_id: 1,
          medoid: 'image_012.png',
          distance: 0.12,
        },
      }],
    };
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.visualResults?.stimuli[0].projection?.nearestReferenceMembers[0]).toEqual({
      imageName: 'image_012.png',
      imageUrl: '/images/image_012.png',
      distance: 0.08,
    });
    expect(session.visualResults?.stimuli[0].hierarchicalAssignment?.medoid.imageName).toBe('image_012.png');
  });

  it('reports expired sessions distinctly', () => {
    const raw = rawSessionFixture();
    raw.expires_at = '2020-01-01T00:00:00Z';
    try {
      validateComparisonSession(raw, 'http://core.example:5556', new Date('2021-01-01T00:00:00Z'));
      throw new Error('expected validation failure');
    } catch (error) {
      expect(error).toBeInstanceOf(CortexSessionError);
      expect((error as CortexSessionError).kind).toBe('expired');
    }
  });

  it('rejects reference-version mismatches', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const rawManifest = rawReferenceManifestFixture();
    rawManifest.reference_version = 'different-reference';
    const manifest = validateReferenceManifest(rawManifest);

    expect(() => assertSessionReferenceCompatible(session, manifest)).toThrowError(/different historical reference/i);
  });
});

describe('session URL parsing', () => {
  it('leaves direct static visits in static mode', () => {
    expect(extractSessionToken('')).toBeNull();
    expect(extractSessionToken('#tab=visualization')).toBeNull();
  });

  it('accepts an opaque fragment token and rejects path-like values', () => {
    expect(extractSessionToken('#session=abcdefghijklmnop_123')).toBe('abcdefghijklmnop_123');
    expect(() => extractSessionToken('#session=../../local/path')).toThrowError(/malformed/i);
  });
});
