export const CANONICAL_ROIS = ['ffa', 'ppa', 'eba'] as const;

export type CanonicalRoi = (typeof CANONICAL_ROIS)[number];

export type ReferenceNeighbor = {
  imageName: string;
  imageUrl: string;
  value: number;
  rank: number;
  distance?: number;
};

export type VoxelProjection = {
  x: number;
  y: number;
  clusterId: string;
  centroidDistance: number;
  nearestReferenceMembers: ReferenceNeighbor[];
};

export type HierarchicalAssignment = {
  method: 'nearest-medoid assignment';
  clusterId: string;
  medoid: ReferenceNeighbor;
  distance: number;
};

export type JointHierarchyMember = {
  kind: 'reference' | 'uploaded';
  id: string;
  clusterId: string;
  distanceToMedoid: number;
};

export type JointHierarchy = {
  schemaVersion: 1;
  method: 'hierarchical_correlation';
  distanceMetric: 'correlation';
  linkageMethod: 'average';
  normalization: 'reference_zscore';
  referenceCount: number;
  uploadedCount: number;
  clusterCount: number;
  silhouette: number | null;
  members: JointHierarchyMember[];
  linkage: [number, number, number, number][];
  medoidIndices: number[];
};

export type SessionStimulus = {
  id: string;
  displayName: string;
  imageUrl: string;
  color: string;
};

export type StimulusRoiResult = {
  stimulusId: string;
  mean: number;
  sem: number | null;
  referenceRank: number;
  referencePercentile: number;
  sessionRank: number;
  sessionCount: number;
  predecessor: ReferenceNeighbor | null;
  successor: ReferenceNeighbor | null;
  closestByMean: ReferenceNeighbor[];
  closestInVoxelSpace: ReferenceNeighbor[];
  entersTop: boolean;
  entersBottom: boolean;
  voxelProjection: VoxelProjection;
  hierarchicalAssignment: HierarchicalAssignment;
};

export type SessionRoiResult = {
  stimuli: StimulusRoiResult[];
  topReferences: ReferenceNeighbor[];
  bottomReferences: ReferenceNeighbor[];
  jointHierarchy?: JointHierarchy;
};

export type FeatureReferenceNeighbor = {
  imageName: string;
  imageUrl: string;
  distance: number;
};

export type FeatureProjection = {
  x: number;
  y: number;
  clusterId: string;
  centroidDistance: number;
  nearestReferenceMembers: FeatureReferenceNeighbor[];
};

export type FeatureHierarchicalAssignment = {
  method: 'nearest-medoid assignment';
  clusterId: string;
  medoid: FeatureReferenceNeighbor;
  distance: number;
};

export type OptionalProjectionStimulus = {
  stimulusId: string;
  projection: FeatureProjection | null;
  hierarchicalAssignment: FeatureHierarchicalAssignment | null;
};

export type OptionalProjectionResults = {
  stimuli: OptionalProjectionStimulus[];
  jointHierarchy?: JointHierarchy;
};

export const SUBJECT_ONE_ROI_VOXEL_COUNTS: Record<CanonicalRoi, number> = {
  ffa: 156,
  ppa: 170,
  eba: 518,
};

export type BrainMapStimulusResult = {
  stimulusId: string;
  roiResponses: Record<CanonicalRoi, number[]>;
};

export type BrainMapResults = {
  schemaVersion: 1;
  subjectId: 's1';
  responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  roiVoxelCounts: Record<CanonicalRoi, number>;
  stimuli: BrainMapStimulusResult[];
};

export type SessionModelResults = {
  roiResults: Record<CanonicalRoi, SessionRoiResult>;
  brainMapResults?: BrainMapResults;
};

export type CortexComparisonSession = {
  schemaVersion: 1;
  createdAt: string;
  expiresAt: string;
  sourceRequest: {
    model: string;
    dataset: string;
    roi: string;
  };
  comparison: {
    model: string;
    dataset: 'nsd_1000';
    rois: CanonicalRoi[];
    referenceVersion: string;
    referenceCount: 185;
  };
  stimuli: SessionStimulus[];
  roiResults: Record<CanonicalRoi, SessionRoiResult>;
  visualResults?: OptionalProjectionResults;
  dreamsimResults?: OptionalProjectionResults;
  brainMapResults?: BrainMapResults;
  /** Additional models evaluated on the identical uploads and reference release. */
  modelResults?: Record<string, SessionModelResults>;
};

export type PublicReferenceManifest = {
  schemaVersion: number;
  referenceVersion: string;
  dataset: 'nsd_1000';
  referenceCount: number;
  models: string[];
  rois: CanonicalRoi[];
};

export type SessionLoadErrorKind =
  | 'configuration'
  | 'invalid'
  | 'expired'
  | 'unavailable'
  | 'version-mismatch';

export type SessionLoadState =
  | { status: 'static' }
  | { status: 'loading'; token: string }
  | { status: 'ready'; session: CortexComparisonSession }
  | { status: 'error'; kind: SessionLoadErrorKind; message: string };

export type SessionClusterOverlay = {
  stimulusId: string;
  displayName: string;
  imageUrl: string;
  color: string;
  x: number;
  y: number;
  clusterId: string;
  distance: number;
  assignmentMethod: 'nearest-centroid assignment' | 'nearest-medoid assignment' | 'joint hierarchical clustering';
  medoidImageName?: string;
};
