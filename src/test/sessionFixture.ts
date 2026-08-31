const neighbor = (rank: number) => ({
  image_name: `image_${String(rank).padStart(3, '0')}.png`,
  image_url: `/images/image_${String(rank).padStart(3, '0')}.png`,
  value: 1 - rank / 100,
  rank,
  distance: rank / 100,
});

const stimulusResult = (roiOffset: number) => ({
  stimulus_id: 'stimulus-a',
  mean: 0.8 + roiOffset,
  sem: 0.03,
  reference_rank: 5.5,
  reference_percentile: 97.5,
  session_rank: 6,
  session_count: 186,
  predecessor: neighbor(5),
  successor: neighbor(6),
  closest_by_mean: [neighbor(5), neighbor(6)],
  closest_in_voxel_space: [neighbor(7)],
  enters_top: true,
  enters_bottom: false,
  voxel_projection: {
    x: 1 + roiOffset,
    y: -2 + roiOffset,
    cluster_id: 2,
    centroid_distance: 0.42,
    nearest_reference_members: [neighbor(7)],
  },
  hierarchical_assignment: {
    method: 'nearest-medoid assignment',
    cluster_id: 2,
    medoid: neighbor(8),
    distance: 0.23,
  },
});

export function rawSessionFixture() {
  return {
    schema_version: 1,
    created_at: '2099-01-01T00:00:00Z',
    expires_at: '2099-01-01T01:00:00Z',
    source_request: {
      model: 'clip_rn50',
      dataset: 'murty185',
      roi: 'eba',
    },
    comparison: {
      model: 'clip_rn50',
      dataset: 'nsd_1000',
      rois: ['ffa', 'ppa', 'eba'],
      reference_version: 'reference-v1',
      reference_count: 185,
    },
    stimuli: [{
      id: 'stimulus-a',
      display_name: 'uploaded.png',
      image_url: '/comparison-sessions/opaque-token/images/stimulus-a',
      color: '#d946ef',
    }],
    roi_results: {
      ffa: { stimuli: [stimulusResult(0)], top_references: [neighbor(1)], bottom_references: [neighbor(185)] },
      ppa: { stimuli: [stimulusResult(0.1)], top_references: [neighbor(1)], bottom_references: [neighbor(185)] },
      eba: { stimuli: [stimulusResult(0.2)], top_references: [neighbor(1)], bottom_references: [neighbor(185)] },
    },
  };
}

export function rawReferenceManifestFixture() {
  return {
    schema_version: 1,
    reference_version: 'reference-v1',
    dataset: 'nsd_1000',
    reference_count: 185,
    models: ['clip_rn50'],
    rois: ['ffa', 'ppa', 'eba'],
  };
}
