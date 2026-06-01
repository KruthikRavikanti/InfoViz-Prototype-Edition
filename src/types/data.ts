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
  pcaDimUsed?: number | null;
  bestK: number | null;
  silhouette: number | null;
  clusterSizes: Record<string, number>;
  explainedVarianceRatioFirst10?: number[];
  featureSet?: string;
  projectionUsedForClustering?: string;
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

export type ClusteringData = {
  voxelSummaries: ClusterSummary[];
  visualSummary: ClusterSummary | null;
  visualView: ClusterView | null;
  imageCategories: Map<string, ImageCategory>;
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

export type CompareSummary = {
  top: ImageSetComparison;
  bottom: ImageSetComparison;
  aggregateScoreDifference: number | null;
  imageMeanDifference: number | null;
};
