export type Roi = 'ffa' | 'ppa' | 'eba' | 'Overall' | (string & {});

export type WideCsvRow = {
  image_name: string;
} & Record<string, string | number | undefined>;

export type ModelRoiColumn = {
  columnName: string;
  model: string;
  roi: Roi;
};

export type LongFormDatum = {
  imageName: string;
  imageUrl: string;
  model: string;
  roi: Roi;
  value: number;
  columnName: string;
};

export type JsonScoreObject = Record<Roi, Record<string, number>>;

export type VisualizationData = {
  rows: WideCsvRow[];
  scores: JsonScoreObject;
  modelRoiColumns: ModelRoiColumn[];
  clustering: ClusteringData;
};

export type ClusteringMethod = 'kmeans_euclidean' | 'hierarchical_correlation';
export type VoxelClusteringMethod = ClusteringMethod;
export type RankingSystem = 'overall' | 'roi';
export type SortDirection = 'desc' | 'asc';

export type AggregateHeatmapCell = {
  id: string;
  roi: Roi;
  model: string;
  score: number | null;
  rankWithinRoi: number | null;
  overallScore: number | null;
};

export type SelectedHeatmapCell = AggregateHeatmapCell;

export type EvidenceImage = {
  imageName: string;
  imageUrl: string;
  value: number;
  valueLabel?: string;
  rank: number;
};

export type ClusterSummary = {
  sourceFile?: string;
  resultCsv?: string;
  model?: string;
  roi?: Roi;
  nImages: number | null;
  nVoxels?: number | null;
  nFeatures?: number | null;
  embeddingKey?: string;
  pcaDimUsed?: number | null;
  bestK: number | null;
  silhouette: number | null;
  clusterSizes: Record<string, number>;
  explainedVarianceRatioFirst10?: number[];
  featureSet?: string;
  projectionUsedForClustering?: string;
  clusteringMethod?: string;
  distanceMetric?: string;
  linkageMethod?: string;
  normalization?: string;
  linkageCsv?: string;
  dendrogramPng?: string;
  distanceToClusterRepresentative?: string;
  filePath?: string;
};

export type ClusterPoint = {
  imageName: string;
  imageUrl: string;
  clusterLabel: string;
  value: number;
  rank: number;
  pcaX: number | null;
  pcaY: number | null;
  tsneX: number | null;
  tsneY: number | null;
  plotX: number | null;
  plotY: number | null;
  distanceToCentroid: number | null;
};

export type ClusterGroup = {
  label: string;
  size: number;
  images: EvidenceImage[];
  points: ClusterPoint[];
};

export type ClusterView = {
  summary: ClusterSummary | null;
  points: ClusterPoint[];
  groups: ClusterGroup[];
  coordinateMode: 'plot' | 'pca' | 'tsne' | null;
};

export type ImageCategory = 'Faces' | 'Places' | 'Body Part';

export type GroundTruthRoiView = {
  roi: string;
  view: ClusterView;
};

export type GroundTruthPatient = {
  id: string;       // e.g. "p1", "p2", "dp"
  label: string;    // e.g. "Patient 1", "Patient 2", "DP"
  rois: GroundTruthRoiView[];
};

export type ClusteringData = {
  voxelSummaries: ClusterSummary[];
  voxelSummariesByMethod: Record<ClusteringMethod, ClusterSummary[]>;
  visualSummary: ClusterSummary | null;
  visualView: ClusterView | null;
  visualSummaryByMethod: Record<ClusteringMethod, ClusterSummary | null>;
  visualViewByMethod: Record<ClusteringMethod, ClusterView | null>;
  dreamsimSummary: ClusterSummary | null;
  dreamsimView: ClusterView | null;
  dreamsimSummaryByMethod: Record<ClusteringMethod, ClusterSummary | null>;
  dreamsimViewByMethod: Record<ClusteringMethod, ClusterView | null>;
  imageCategories: Map<string, ImageCategory>;
  groundTruthPatients: GroundTruthPatient[];
  groundTruthPatientsByMethod: Record<ClusteringMethod, GroundTruthPatient[]>;
};

export type EvidenceStats = {
  max: number;
  min: number;
  mean: number;
  median: number;
  standardDeviation: number;
};

export type EvidenceView = {
  columnName: string;
  images: EvidenceImage[];
  topImages: EvidenceImage[];
  bottomImages: EvidenceImage[];
  stats: EvidenceStats;
};

export type ImageSetComparison = {
  overlap: EvidenceImage[];
  uniqueA: EvidenceImage[];
  uniqueB: EvidenceImage[];
};

export type RankSimilaritySummary = {
  spearmanRho: number | null;
  similarityScore: number | null;
  sharedImageCount: number;
};

export type ClusterIntersection = {
  aLabel: string;
  bLabel: string;
  count: number;
};

export type ClusterComparisonSummary = {
  adjustedRandIndex: number | null;
  normalizedMutualInformation: number | null;
  pairAgreement: number | null;
  coClusterJaccard: number | null;
  sharedImageCount: number;
  clusterCountA: number;
  clusterCountB: number;
  intersections: ClusterIntersection[];
  largestIntersections: ClusterIntersection[];
};

export type CompareSummary = {
  top: ImageSetComparison;
  bottom: ImageSetComparison;
  aggregateScoreDifference: number | null;
  imageMeanDifference: number | null;
  rankSimilarity: RankSimilaritySummary;
};
