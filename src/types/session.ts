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
};
