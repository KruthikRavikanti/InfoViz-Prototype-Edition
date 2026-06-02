import { csv, json } from 'd3';
import type { ClusterPoint, ClusterSummary, ClusterView, ClusteringData, EvidenceImage, GroundTruthPatient, GroundTruthRoiView, ImageCategory, Roi, SelectedHeatmapCell } from '../types/data';

const VOXEL_SUMMARY_PATH = '/data/voxel_clusters/voxel_cluster_summary.json';
const VISUAL_SUMMARY_PATH = '/data/visual_clusters/visual_cluster_summary.json';
const VISUAL_CLUSTERS_PATH = '/data/visual_clusters/visual_clusters.csv';
const IMAGE_CATEGORIES_PATH = '/data/murty185Classification.csv';
const GT_BASE = '/data/ground_truth_clusters';
const IMAGE_BASE_PATH = '/images';

// All patient/ROI combos that have files in ground_truth_clusters/
const GT_ENTRIES: Array<{ patient: string; roi: string }> = [
  { patient: 'dp', roi: 'leba' }, { patient: 'dp', roi: 'lffa' },
  { patient: 'dp', roi: 'reba' }, { patient: 'dp', roi: 'rffa' },
  { patient: 'p1', roi: 'leba' }, { patient: 'p1', roi: 'lffa' }, { patient: 'p1', roi: 'lppa' },
  { patient: 'p1', roi: 'reba' }, { patient: 'p1', roi: 'rffa' }, { patient: 'p1', roi: 'rppa' },
  { patient: 'p2', roi: 'leba' }, { patient: 'p2', roi: 'lffa' }, { patient: 'p2', roi: 'lppa' },
  { patient: 'p2', roi: 'reba' }, { patient: 'p2', roi: 'rffa' }, { patient: 'p2', roi: 'rppa' },
  { patient: 'p3', roi: 'leba' }, { patient: 'p3', roi: 'lffa' }, { patient: 'p3', roi: 'lppa' },
  { patient: 'p3', roi: 'reba' }, { patient: 'p3', roi: 'rffa' }, { patient: 'p3', roi: 'rppa' },
  { patient: 'p4', roi: 'leba' }, { patient: 'p4', roi: 'lffa' }, { patient: 'p4', roi: 'lppa' },
  { patient: 'p4', roi: 'reba' }, { patient: 'p4', roi: 'rffa' }, { patient: 'p4', roi: 'rppa' },
];

function patientLabel(id: string): string {
  if (id === 'dp') return 'DP';
  const num = id.replace(/^p/, '');
  return `Patient ${num}`;
}

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

function parseClusterPoint(row: RawClusterRow, index: number, nameTransform?: (n: string) => string): ClusterPoint | null {
  if (!row.image_name) {
    return null;
  }

  const rawName = String(row.image_name);
  const imageName = nameTransform ? nameTransform(rawName) : rawName;
  const distanceToCentroid = toFiniteNumber(row.distance_to_centroid);

  return {
    imageName,
    imageUrl: `${IMAGE_BASE_PATH}/${imageName}`,
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
    .sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));

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

// Ground truth CSVs use 4-digit zero-padded names without extension: image_0001 → image_001.png
function normalizeGtImageName(raw: string): string {
  const match = raw.match(/^image_0*(\d+)$/);
  if (!match) return raw;
  const num = Number(match[1]);
  return `image_${String(num).padStart(3, '0')}.png`;
}

async function loadGroundTruthPatients(): Promise<GroundTruthPatient[]> {
  const results = await Promise.allSettled(
    GT_ENTRIES.map(async ({ patient, roi }) => {
      const name = `${patient}__${roi}`;
      const [rawSummary, points] = await Promise.all([
        json<RawClusterSummary>(`${GT_BASE}/${name}_summary.json`),
        csv(`${GT_BASE}/${name}_clusters.csv`, (row, idx) =>
          parseClusterPoint(row as RawClusterRow, idx, normalizeGtImageName),
        ).then((rows) => rows.filter((p): p is ClusterPoint => p !== null)),
      ]);
      const summary = rawSummary ? normalizeSummary(rawSummary) : null;
      const view = buildClusterView(summary, points);
      return { patient, roi, view };
    }),
  );

  const byPatient = new Map<string, GroundTruthRoiView[]>();
  for (const result of results) {
    if (result.status !== 'fulfilled' || !result.value.view) continue;
    const { patient, roi, view } = result.value;
    const rois = byPatient.get(patient) ?? [];
    rois.push({ roi, view });
    byPatient.set(patient, rois);
  }

  const patientOrder = ['dp', 'p1', 'p2', 'p3', 'p4'];
  return patientOrder
    .filter((id) => byPatient.has(id))
    .map((id) => ({
      id,
      label: patientLabel(id),
      rois: byPatient.get(id)!,
    }));
}

export async function loadClusteringData(): Promise<ClusteringData> {
  const [voxelSummaries, visualClusters, imageCategories, groundTruthPatients] = await Promise.all([
    loadVoxelSummaries(),
    loadVisualClusterView(),
    loadImageCategories(),
    loadGroundTruthPatients(),
  ]);

  return {
    voxelSummaries,
    visualSummary: visualClusters.summary,
    visualView: visualClusters.view,
    imageCategories,
    groundTruthPatients,
  };
}
