import { csv, json } from 'd3';
import type { ClusterPoint, ClusterSummary, ClusterView, ClusteringData, EvidenceImage, ImageCategory, Roi, SelectedHeatmapCell } from '../types/data';

const VOXEL_SUMMARY_PATH = '/data/voxel_clusters/voxel_cluster_summary.json';
const VISUAL_SUMMARY_PATH = '/data/visual_clusters/visual_cluster_summary.json';
const VISUAL_CLUSTERS_PATH = '/data/visual_clusters/visual_clusters.csv';
const IMAGE_CATEGORIES_PATH = '/data/murty185Classification.csv';
const IMAGE_BASE_PATH = '/images';

type RawClusterSummary = {
  source_file?: string;
  result_csv?: string;
  model?: string;
  roi?: Roi;
  n_images?: number;
  n_voxels?: number;
  pca_dim_used?: number;
  best_k?: number;
  silhouette?: number;
  cluster_sizes?: Record<string, number>;
  explained_variance_ratio_first10?: number[];
  feature_set?: string;
  projection_used_for_clustering?: string;
};

type RawClusterRow = {
  image_name?: string;
  cluster_label?: string | number;
  pca_x?: string | number;
  pca_y?: string | number;
  tsne_x?: string | number;
  tsne_y?: string | number;
  plot_x?: string | number;
  plot_y?: string | number;
  distance_to_centroid?: string | number;
};

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (value === null || value === undefined || value === '') {
    return null;
  }

  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function normalizeSummary(summary: RawClusterSummary): ClusterSummary {
  return {
    sourceFile: summary.source_file,
    resultCsv: summary.result_csv,
    model: summary.model,
    roi: summary.roi,
    nImages: toFiniteNumber(summary.n_images),
    nVoxels: toFiniteNumber(summary.n_voxels),
    pcaDimUsed: toFiniteNumber(summary.pca_dim_used),
    bestK: toFiniteNumber(summary.best_k),
    silhouette: toFiniteNumber(summary.silhouette),
    clusterSizes: summary.cluster_sizes ?? {},
    explainedVarianceRatioFirst10: Array.isArray(summary.explained_variance_ratio_first10)
      ? summary.explained_variance_ratio_first10.filter((value) => Number.isFinite(value))
      : undefined,
    featureSet: summary.feature_set,
    projectionUsedForClustering: summary.projection_used_for_clustering,
  };
}

function deriveVoxelClusterPath(summary: ClusterSummary): string | null {
  if (!summary.sourceFile) {
    return null;
  }

  return `/data/voxel_clusters/${summary.sourceFile.replace(/\.csv$/i, '_clusters.csv')}`;
}

function parseClusterPoint(row: RawClusterRow, index: number): ClusterPoint | null {
  if (!row.image_name) {
    return null;
  }

  const distanceToCentroid = toFiniteNumber(row.distance_to_centroid);

  return {
    imageName: String(row.image_name),
    imageUrl: `${IMAGE_BASE_PATH}/${row.image_name}`,
    clusterLabel: String(row.cluster_label ?? 'unassigned'),
    value: distanceToCentroid ?? 0,
    rank: index + 1,
    pcaX: toFiniteNumber(row.pca_x),
    pcaY: toFiniteNumber(row.pca_y),
    tsneX: toFiniteNumber(row.tsne_x),
    tsneY: toFiniteNumber(row.tsne_y),
    plotX: toFiniteNumber(row.plot_x),
    plotY: toFiniteNumber(row.plot_y),
    distanceToCentroid,
  };
}

function pointToImage(point: ClusterPoint): EvidenceImage {
  return {
    imageName: point.imageName,
    imageUrl: point.imageUrl,
    value: point.value,
    valueLabel: point.distanceToCentroid === null ? `Cluster ${point.clusterLabel}` : `Distance ${point.distanceToCentroid.toFixed(2)}`,
    rank: point.rank,
  };
}

function inferCoordinateMode(points: ClusterPoint[]): ClusterView['coordinateMode'] {
  if (points.some((point) => point.plotX !== null && point.plotY !== null)) {
    return 'plot';
  }

  if (points.some((point) => point.pcaX !== null && point.pcaY !== null)) {
    return 'pca';
  }

  if (points.some((point) => point.tsneX !== null && point.tsneY !== null)) {
    return 'tsne';
  }

  return null;
}

export function buildClusterView(summary: ClusterSummary | null, points: ClusterPoint[]): ClusterView | null {
  if (!summary && points.length === 0) {
    return null;
  }

  const groupsByLabel = new Map<string, ClusterPoint[]>();

  for (const point of points) {
    const group = groupsByLabel.get(point.clusterLabel) ?? [];
    group.push(point);
    groupsByLabel.set(point.clusterLabel, group);
  }

  const groups = Array.from(groupsByLabel.entries())
    .map(([label, groupPoints]) => ({
      label,
      size: summary?.clusterSizes[label] ?? groupPoints.length,
      points: groupPoints,
      images: groupPoints
        .slice()
        .sort((a, b) => (a.distanceToCentroid ?? Number.POSITIVE_INFINITY) - (b.distanceToCentroid ?? Number.POSITIVE_INFINITY))
        .slice(0, 6)
        .map(pointToImage),
    }))
    .sort((a, b) => b.size - a.size || a.label.localeCompare(b.label, undefined, { numeric: true }));

  return {
    summary,
    points,
    groups,
    coordinateMode: inferCoordinateMode(points),
  };
}

export function findVoxelClusterSummary(
  voxelSummaries: ClusterSummary[],
  selectedCell: SelectedHeatmapCell,
): ClusterSummary | null {
  return voxelSummaries.find((summary) => summary.model === selectedCell.model && summary.roi === selectedCell.roi) ?? null;
}

export async function loadVoxelClusterView(
  voxelSummaries: ClusterSummary[],
  selectedCell: SelectedHeatmapCell,
): Promise<ClusterView | null> {
  const summary = findVoxelClusterSummary(voxelSummaries, selectedCell);

  if (!summary) {
    return null;
  }

  const filePath = summary.filePath ?? deriveVoxelClusterPath(summary);

  if (!filePath) {
    return buildClusterView(summary, []);
  }

  const points = await csv(filePath, (row, index) => parseClusterPoint(row as RawClusterRow, index)).then((rows) =>
    rows.filter((point): point is ClusterPoint => point !== null),
  );

  return buildClusterView(summary, points);
}

async function loadVoxelSummaries(): Promise<ClusterSummary[]> {
  try {
    const summaries = await json<RawClusterSummary[]>(VOXEL_SUMMARY_PATH);

    if (!Array.isArray(summaries)) {
      return [];
    }

    return summaries.map((summary) => {
      const normalized = normalizeSummary(summary);
      return {
        ...normalized,
        filePath: deriveVoxelClusterPath(normalized) ?? undefined,
      };
    });
  } catch {
    return [];
  }
}

async function loadVisualClusterView(): Promise<{ summary: ClusterSummary | null; view: ClusterView | null }> {
  try {
    const [rawSummary, points] = await Promise.all([
      json<RawClusterSummary>(VISUAL_SUMMARY_PATH),
      csv(VISUAL_CLUSTERS_PATH, (row, index) => parseClusterPoint(row as RawClusterRow, index)).then((rows) =>
        rows.filter((point): point is ClusterPoint => point !== null),
      ),
    ]);
    const summary = rawSummary ? normalizeSummary(rawSummary) : null;

    return {
      summary,
      view: buildClusterView(summary, points),
    };
  } catch {
    return {
      summary: null,
      view: null,
    };
  }
}

async function loadImageCategories(): Promise<Map<string, ImageCategory>> {
  const map = new Map<string, ImageCategory>();
  try {
    const text = await fetch(IMAGE_CATEGORIES_PATH).then((r) => r.text());
    for (const line of text.split('\n')) {
      const comma = line.indexOf(',');
      if (comma === -1) continue;
      const name = line.slice(0, comma).trim();
      const cat = line.slice(comma + 1).trim() as ImageCategory;
      if (name) map.set(name, cat);
    }
  } catch {
    // non-fatal — scatter just won't have category highlighting
  }
  return map;
}

export async function loadClusteringData(): Promise<ClusteringData> {
  const [voxelSummaries, visualClusters, imageCategories] = await Promise.all([
    loadVoxelSummaries(),
    loadVisualClusterView(),
    loadImageCategories(),
  ]);

  return {
    voxelSummaries,
    visualSummary: visualClusters.summary,
    visualView: visualClusters.view,
    imageCategories,
  };
}
