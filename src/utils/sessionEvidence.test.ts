import { describe, expect, it } from 'vitest';
import type { EvidenceImage, JsonScoreObject } from '../types/data';
import { buildAggregateHeatmapCells } from './heatmap';
import { buildOptionalSessionOverlays, buildVoxelSessionOverlays, mergeReferenceAndSessionRanks } from './sessionEvidence';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import type { OptionalProjectionResults } from '../types/session';
import { jointHierarchyFixture } from '../test/jointHierarchyFixture';

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

describe('fixed-reference upload cluster overlays', () => {
  function sessionWithDifferentAssignments() {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5558');
    session.roiResults.eba.stimuli[0].hierarchicalAssignment.clusterId = '9';
    session.roiResults.eba.jointHierarchy = jointHierarchyFixture();
    return session;
  }

  it('keeps existing k-means positions and centroid metadata unchanged', () => {
    const session = sessionWithDifferentAssignments();
    const [overlay] = buildVoxelSessionOverlays(session, 'eba');
    expect(overlay).toMatchObject({
      x: 1.2, y: -1.8, clusterId: '2', distance: .42,
      assignmentMethod: 'nearest-centroid assignment', color: '#1c1917',
    });
    expect(overlay.medoidImageName).toBeUndefined();
  });

  it('uses joint membership, not nearest-medoid assignment, with fixed verified axes', () => {
    const session = sessionWithDifferentAssignments();
    const original = structuredClone(session);
    const [overlay] = buildVoxelSessionOverlays(session, 'eba', 'hierarchical_correlation', { xSign: 1, ySign: -1 });
    expect(overlay).toMatchObject({
      x: 1.2, y: 1.8, clusterId: '1', distance: .23,
      assignmentMethod: 'joint hierarchical clustering', medoidImageName: 'reference-b.png',
      color: '#1c1917',
    });
    expect(session).toEqual(original);
  });

  it('does not draw hierarchical uploads when the axes are unchecked', () => {
    const session = sessionWithDifferentAssignments();
    expect(buildVoxelSessionOverlays(session, 'eba', 'hierarchical_correlation')).toEqual([]);
    expect(buildVoxelSessionOverlays(session, 'eba', 'hierarchical_correlation', null)).toEqual([]);
  });

  const visualResults: OptionalProjectionResults = {
    jointHierarchy: jointHierarchyFixture(),
    stimuli: [{
      stimulusId: 'stimulus-a',
      projection: { x: 3, y: -7, clusterId: '2', centroidDistance: 9, nearestReferenceMembers: [] },
      hierarchicalAssignment: {
        method: 'nearest-medoid assignment', clusterId: '8', distance: .3,
        medoid: { imageName: 'image_004.png', imageUrl: '/images/image_004.png', distance: .3 },
      },
    }],
  };

  it('uses optional visual joint membership and the verified x flip', () => {
    const session = sessionWithDifferentAssignments();
    const original = structuredClone(visualResults);
    const [overlay] = buildOptionalSessionOverlays(session, visualResults, 'hierarchical_correlation', { xSign: -1, ySign: 1 });
    expect(overlay).toMatchObject({
      x: -3, y: -7, clusterId: '1', distance: .23,
      assignmentMethod: 'joint hierarchical clustering', medoidImageName: 'reference-b.png',
    });
    expect(visualResults).toEqual(original);
  });

  it('skips optional hierarchical uploads with missing projection, joint hierarchy, or alignment', () => {
    const session = sessionWithDifferentAssignments();
    const signs = { xSign: 1, ySign: 1 } as const;
    expect(buildOptionalSessionOverlays(session, undefined, 'hierarchical_correlation', signs)).toEqual([]);
    expect(buildOptionalSessionOverlays(session, visualResults, 'hierarchical_correlation')).toEqual([]);
    const withoutProjection = structuredClone(visualResults);
    withoutProjection.stimuli[0].projection = null;
    expect(buildOptionalSessionOverlays(session, withoutProjection, 'hierarchical_correlation', signs)).toEqual([]);
    const withoutJoint = structuredClone(visualResults);
    delete withoutJoint.jointHierarchy;
    expect(buildOptionalSessionOverlays(session, withoutJoint, 'hierarchical_correlation', signs)).toEqual([]);
  });

  it('does not silently fall back to approximate assignments for legacy hierarchical sessions', () => {
    const session = sessionWithDifferentAssignments();
    delete session.roiResults.eba.jointHierarchy;
    expect(buildVoxelSessionOverlays(session, 'eba', 'hierarchical_correlation', { xSign: 1, ySign: 1 })).toEqual([]);
  });

  it('can describe a cluster whose medoid is itself an uploaded image', () => {
    const session = sessionWithDifferentAssignments();
    session.roiResults.eba.jointHierarchy!.medoidIndices[1] = 2;
    session.roiResults.eba.jointHierarchy!.members[2].distanceToMedoid = 0;
    const [overlay] = buildVoxelSessionOverlays(session, 'eba', 'hierarchical_correlation', { xSign: 1, ySign: 1 });
    expect(overlay.medoidImageName).toBe('uploaded.png');
  });
});
