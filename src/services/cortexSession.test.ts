import { describe, expect, it } from 'vitest';
import {
  CortexSessionError,
  assertSessionReferenceCompatible,
  extractSessionToken,
  resolveCortexUrl,
  validateComparisonSession,
  validateReferenceManifest,
} from './cortexSession';
import { rawJointHierarchyFixture, rawReferenceManifestFixture, rawSessionFixture } from '../test/sessionFixture';

function rawJointSessionFixture(uploadedIds = ['stimulus-a']) {
  const raw = rawSessionFixture();
  const stimuli = uploadedIds.map((id) => ({
    ...raw.stimuli[0],
    id,
    image_url: `/comparison-sessions/opaque-token/images/${id}`,
  }));
  const roiResult = (roi: 'ffa' | 'ppa' | 'eba') => ({
    ...raw.roi_results[roi],
    stimuli: uploadedIds.map((id, index) => ({
      ...raw.roi_results[roi].stimuli[0],
      stimulus_id: id,
      session_rank: index + 6,
      session_count: 185 + uploadedIds.length,
    })),
    joint_hierarchy: rawJointHierarchyFixture(uploadedIds),
  });
  return {
    ...raw,
    stimuli,
    roi_results: { ffa: roiResult('ffa'), ppa: roiResult('ppa'), eba: roiResult('eba') },
    visual_results: {
      stimuli: uploadedIds.map((id) => ({
        stimulus_id: id,
        projection: raw.roi_results.ffa.stimuli[0].voxel_projection,
        hierarchical_assignment: null,
      })),
      joint_hierarchy: rawJointHierarchyFixture(uploadedIds),
    },
  };
}

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

  it('parses a complete NSD Subject 1 brain-map response payload', () => {
    const raw = rawSessionFixture() as ReturnType<typeof rawSessionFixture> & { brain_map_results?: unknown };
    raw.brain_map_results = {
      schema_version: 1,
      subject_id: 's1',
      response_space: 'nsd_1000_subject1_floc_t7_v1',
      roi_voxel_counts: { ffa: 156, ppa: 170, eba: 518 },
      stimuli: [{
        stimulus_id: 'stimulus-a',
        roi_responses: {
          ffa: Array.from({ length: 156 }, (_, index) => index / 100),
          ppa: Array.from({ length: 170 }, (_, index) => -index / 100),
          eba: Array.from({ length: 518 }, () => 0),
        },
      }],
    };

    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.brainMapResults?.subjectId).toBe('s1');
    expect(session.brainMapResults?.stimuli[0].roiResponses.eba).toHaveLength(518);
  });

  it('rejects brain-map responses that cannot preserve the Subject 1 voxel order', () => {
    const raw = rawSessionFixture() as ReturnType<typeof rawSessionFixture> & { brain_map_results?: unknown };
    raw.brain_map_results = {
      schema_version: 1,
      subject_id: 's1',
      response_space: 'nsd_1000_subject1_floc_t7_v1',
      roi_voxel_counts: { ffa: 156, ppa: 170, eba: 518 },
      stimuli: [{
        stimulus_id: 'stimulus-a',
        roi_responses: {
          ffa: Array.from({ length: 155 }, () => 0),
          ppa: Array.from({ length: 170 }, () => 0),
          eba: Array.from({ length: 518 }, () => 0),
        },
      }],
    };

    expect(() => validateComparisonSession(raw, 'http://core.example:5556')).toThrowError(
      /must contain 156 responses/i,
    );
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

describe('joint hierarchical clustering contract', () => {
  it('normalizes complete ROI and visual hierarchies without replacing legacy assignments', () => {
    const raw = rawJointSessionFixture();
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    const hierarchy = session.roiResults.ffa.jointHierarchy;

    expect(hierarchy).toMatchObject({
      schemaVersion: 1,
      method: 'hierarchical_correlation',
      distanceMetric: 'correlation',
      linkageMethod: 'average',
      normalization: 'reference_zscore',
      referenceCount: 185,
      uploadedCount: 1,
      clusterCount: 2,
      silhouette: 0.8,
      medoidIndices: [0, 93],
    });
    expect(hierarchy?.members).toHaveLength(186);
    expect(hierarchy?.members[185]).toEqual({
      kind: 'uploaded', id: 'stimulus-a', clusterId: '1', distanceToMedoid: 0.2,
    });
    expect(hierarchy?.linkage).toEqual(raw.roi_results.ffa.joint_hierarchy.linkage);
    expect(session.visualResults?.jointHierarchy?.members[185].id).toBe('stimulus-a');
    expect(session.roiResults.ffa.stimuli[0].hierarchicalAssignment.method).toBe('nearest-medoid assignment');
    expect(raw.roi_results.ffa.joint_hierarchy.members[185].cluster_id).toBe(1);
  });

  it('keeps legacy sessions explicitly without joint clustering evidence', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    expect(session.roiResults.ffa.jointHierarchy).toBeUndefined();
    expect(session.roiResults.ppa.jointHierarchy).toBeUndefined();
    expect(session.roiResults.eba.jointHierarchy).toBeUndefined();
  });

  it('accepts reference leaf orders that differ across feature spaces', () => {
    const raw = rawJointSessionFixture();
    raw.visual_results.joint_hierarchy.silhouette = 0.75;
    const referenceIds = raw.visual_results.joint_hierarchy.members.slice(0, 185).map((member) => member.id).reverse();
    raw.visual_results.joint_hierarchy.members.slice(0, 185).forEach((member, index) => {
      member.id = referenceIds[index];
    });
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.visualResults?.jointHierarchy?.silhouette).toBe(0.75);
    expect(session.visualResults?.jointHierarchy?.members[0].id).toBe('image_185.png');
  });

  it('accepts null silhouette only for a single-cluster fallback', () => {
    const raw = rawJointSessionFixture();
    const hierarchy = raw.roi_results.ffa.joint_hierarchy;
    hierarchy.cluster_count = 1;
    hierarchy.silhouette = null;
    hierarchy.members.forEach((member) => { member.cluster_id = 0; });
    hierarchy.medoid_indices = [0];
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.roiResults.ffa.jointHierarchy?.silhouette).toBeNull();
    expect(session.roiResults.ffa.jointHierarchy?.clusterCount).toBe(1);
  });

  it('accepts an uploaded image as a cluster representative', () => {
    const raw = rawJointSessionFixture();
    raw.roi_results.ffa.joint_hierarchy.medoid_indices[1] = 185;
    raw.roi_results.ffa.joint_hierarchy.members[185].distance_to_medoid = 0;
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    expect(session.roiResults.ffa.jointHierarchy?.medoidIndices).toEqual([0, 185]);
  });

  it('requires exact uploaded stimulus order, not just the same set', () => {
    const raw = rawJointSessionFixture(['stimulus-a', 'stimulus-b']);
    const hierarchy = raw.roi_results.ffa.joint_hierarchy;
    hierarchy.members[185].id = 'stimulus-b';
    hierarchy.members[186].id = 'stimulus-a';
    expect(() => validateComparisonSession(raw)).toThrowError(/uploaded stimulus order/i);
  });

  it('rejects mismatched reference sets across otherwise valid hierarchies', () => {
    const raw = rawJointSessionFixture();
    raw.visual_results.joint_hierarchy.members[0].id = 'unknown-reference.png';
    expect(() => validateComparisonSession(raw)).toThrowError(/same reference image IDs/i);
  });

  const malformedCases: [string, (hierarchy: ReturnType<typeof rawJointHierarchyFixture>) => void][] = [
    ['wrong schema version', (hierarchy) => { hierarchy.schema_version = 2; }],
    ['wrong distance metric', (hierarchy) => { hierarchy.distance_metric = 'euclidean'; }],
    ['wrong linkage method', (hierarchy) => { hierarchy.linkage_method = 'ward'; }],
    ['wrong normalization', (hierarchy) => { hierarchy.normalization = 'joint_zscore'; }],
    ['reference count mismatch', (hierarchy) => { hierarchy.reference_count = 184; }],
    ['upload count mismatch', (hierarchy) => { hierarchy.uploaded_count = 0; }],
    ['missing member', (hierarchy) => { hierarchy.members.pop(); }],
    ['duplicate reference ID', (hierarchy) => { hierarchy.members[1].id = hierarchy.members[0].id; }],
    ['reference/upload order violation', (hierarchy) => { hierarchy.members[0].kind = 'uploaded'; }],
    ['unknown uploaded ID', (hierarchy) => { hierarchy.members[185].id = 'unknown-stimulus'; }],
    ['negative cluster ID', (hierarchy) => { hierarchy.members[0].cluster_id = -1; }],
    ['fractional cluster ID', (hierarchy) => { hierarchy.members[0].cluster_id = 0.5; }],
    ['non-contiguous cluster IDs', (hierarchy) => { hierarchy.cluster_count = 3; }],
    ['cluster count beyond searched range', (hierarchy) => { hierarchy.cluster_count = 6; }],
    ['non-finite medoid distance', (hierarchy) => { hierarchy.members[0].distance_to_medoid = NaN; }],
    ['negative medoid distance', (hierarchy) => { hierarchy.members[0].distance_to_medoid = -0.1; }],
    ['medoid distance beyond correlation range', (hierarchy) => { hierarchy.members[0].distance_to_medoid = 2.1; }],
    ['missing tree merge', (hierarchy) => { hierarchy.linkage.pop(); }],
    ['invalid row width', (hierarchy) => { hierarchy.linkage[0] = [0, 1, 0.2] as unknown as [number, number, number, number]; }],
    ['future child node', (hierarchy) => { hierarchy.linkage[0][0] = 186; }],
    ['fractional child node', (hierarchy) => { hierarchy.linkage[0][0] = 0.5; }],
    ['same child used twice', (hierarchy) => { hierarchy.linkage[0][1] = hierarchy.linkage[0][0]; }],
    ['child reused in another merge', (hierarchy) => { hierarchy.linkage[1][1] = 0; }],
    ['incorrect descendant count', (hierarchy) => { hierarchy.linkage[0][3] = 3; }],
    ['non-finite linkage distance', (hierarchy) => { hierarchy.linkage[0][2] = Infinity; }],
    ['linkage distance beyond correlation range', (hierarchy) => { hierarchy.linkage[0][2] = 2.1; }],
    ['non-monotonic linkage distance', (hierarchy) => { hierarchy.linkage[1][2] = 0.19; }],
    ['missing cluster representative', (hierarchy) => { hierarchy.medoid_indices.pop(); }],
    ['representative in another cluster', (hierarchy) => { hierarchy.medoid_indices[0] = 93; }],
    ['representative outside member list', (hierarchy) => { hierarchy.medoid_indices[0] = 186; }],
    ['fractional representative index', (hierarchy) => { hierarchy.medoid_indices[0] = 0.5; }],
    ['representative with non-zero self-distance', (hierarchy) => { hierarchy.members[0].distance_to_medoid = 0.1; }],
    ['non-finite silhouette', (hierarchy) => { hierarchy.silhouette = NaN; }],
    ['silhouette beyond its range', (hierarchy) => { hierarchy.silhouette = 1.1; }],
    ['null multi-cluster silhouette', (hierarchy) => { hierarchy.silhouette = null; }],
    ['finite single-cluster silhouette', (hierarchy) => { hierarchy.cluster_count = 1; }],
  ];

  it.each(malformedCases)('fails closed for %s', (_, mutate) => {
    const raw = rawJointSessionFixture();
    mutate(raw.roi_results.ffa.joint_hierarchy);
    expect(() => validateComparisonSession(raw)).toThrow(CortexSessionError);
  });

  it.each([null, {}, []])('fails closed for a present malformed hierarchy: %j', (malformed) => {
    const raw = rawSessionFixture() as ReturnType<typeof rawSessionFixture> & {
      roi_results: { ffa: { joint_hierarchy?: unknown } };
    };
    raw.roi_results.ffa.joint_hierarchy = malformed;
    expect(() => validateComparisonSession(raw)).toThrow(CortexSessionError);
  });
});

describe('session URL parsing', () => {
  it('routes Core root-relative assets through a same-origin development proxy', () => {
    expect(resolveCortexUrl(
      '/comparison-sessions/token/images/asset',
      '/cortex-core',
      'https://forwarded-preview.example',
    )).toBe('https://forwarded-preview.example/cortex-core/comparison-sessions/token/images/asset');
  });

  it('leaves direct static visits in static mode', () => {
    expect(extractSessionToken('')).toBeNull();
    expect(extractSessionToken('#tab=visualization')).toBeNull();
  });

  it('accepts an opaque fragment token and rejects path-like values', () => {
    expect(extractSessionToken('#session=abcdefghijklmnop_123')).toBe('abcdefghijklmnop_123');
    expect(() => extractSessionToken('#session=../../local/path')).toThrowError(/malformed/i);
  });
});
