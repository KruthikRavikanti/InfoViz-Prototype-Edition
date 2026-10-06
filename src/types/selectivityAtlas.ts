import type { CanonicalRoi } from './session';

export type SelectivityAtlasRoi = CanonicalRoi;

export type SelectivityAtlasPoint = {
  voxelIndex: number;
  roi: SelectivityAtlasRoi;
  roiVoxelIndex: number;
  clusterLabel: number;
  umapX: number;
  umapY: number;
  umapZ: number;
  distanceToCentroid: number;
};

export type SelectivityAtlasStimulus = {
  imageId: string;
  meanZ: number;
};

export type SelectivityAtlasCluster = {
  label: number;
  size: number;
  roiCounts: Record<SelectivityAtlasRoi, number>;
  meanSilhouette: number | null;
  preferredImages: SelectivityAtlasStimulus[];
  suppressedImages: SelectivityAtlasStimulus[];
};

export type SelectivityAtlasMethod = {
  id: string;
  objectClustered: 'voxels';
  normalization: string;
  pcaComponents: number;
  l2Normalization: boolean;
  clustering: string;
  clusterCount: number;
  kmeansNInit: number;
  kmeansRandomState: number;
  embedding: string;
  embeddingUsedForClustering: boolean;
};

export type SelectivityAtlasSummary = {
  schemaVersion: 1;
  model: string;
  dataset: 'nsd_1000';
  stimulusSet: 'murty185';
  subject: 's1';
  responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  responseKind: 'model_predictions';
  embeddingIsAnatomical: false;
  imageCount: number;
  voxelCount: number;
  clusterCount: number;
  silhouette: number;
  pcaExplainedVarianceRatioSum: number;
  splitHalfStability: {
    iterations: number;
    randomSeed: number;
    adjustedRandIndices: number[];
    meanAdjustedRandIndex: number;
    minAdjustedRandIndex: number;
    threshold: number;
    meanPassesThreshold: boolean;
    allPassThreshold: boolean;
  };
  method: SelectivityAtlasMethod;
  k14StabilityCaveat: string;
  sourceFiles: Array<{
    roi: SelectivityAtlasRoi;
    fileName: string;
    sha256: string;
  }>;
  clusters: SelectivityAtlasCluster[];
};

export type SelectivityAtlasManifestModel = {
  pointsUrl: string;
  summaryUrl: string;
  pointsSha256: string;
  summarySha256: string;
  pointCount: number;
};

export type SelectivityAtlasManifest = {
  schemaVersion: 1;
  artifact: 'voxel-selectivity-atlas';
  dataset: 'nsd_1000';
  stimulusSet: 'murty185';
  subject: 's1';
  responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  responseKind: 'model_predictions';
  embeddingIsAnatomical: false;
  imageCount: number;
  voxelCount: number;
  clusterCount: number;
  roiOrder: SelectivityAtlasRoi[];
  rois: Record<SelectivityAtlasRoi, { offset: number; count: number }>;
  method: SelectivityAtlasMethod;
  k14StabilityCaveat: string;
  supportedModels: string[];
  referenceVersion: string;
  imageIdsSha256: string;
  models: Record<string, SelectivityAtlasManifestModel>;
};

export type SelectivityAtlasAssets = {
  manifest: SelectivityAtlasManifest;
  summary: SelectivityAtlasSummary;
  points: SelectivityAtlasPoint[];
};

export type SelectivityAtlasHoverDatum = SelectivityAtlasPoint & {
  canvasX: number;
  canvasY: number;
};
