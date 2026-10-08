import { csvParse } from 'd3';
import type {
  SelectivityAtlasAssets,
  SelectivityAtlasCluster,
  SelectivityAtlasManifest,
  SelectivityAtlasManifestModel,
  SelectivityAtlasMethod,
  SelectivityAtlasPoint,
  SelectivityAtlasRoi,
  SelectivityAtlasStimulus,
  SelectivityAtlasSummary,
} from '../types/selectivityAtlas';
import { publicPath } from '../utils/publicPath';

export const SELECTIVITY_ATLAS_MANIFEST_PATH = publicPath('/data/voxel_selectivity_atlas/manifest.json');

const MANIFEST_SHA256 = '7244cead1cc3a9d6fb0074590eed025889994d23eecce432271cb8b38b28881e';
// The reviewed reference bundle's exact roster; public assets are not JS modules in Vite.
const EXPECTED_MODELS: readonly string[] = [
  'adversarial_inceptionv3', 'aimv2', 'alexnet', 'alexnet_random', 'beit', 'bit', 'blip2',
  'clip_rn101', 'clip_rn101_random', 'clip_rn50', 'clip_rn50_random', 'clip_vit_b32',
  'clip_vit_b32_random', 'convnext', 'cornet_rt', 'cornet_s', 'cornet_z', 'cross_vit',
  'densenet121', 'densenet161', 'densenet169', 'densenet201', 'dinov2_large',
  'dreamsim_vitb16', 'dreamsim_vitb32', 'efficient_net',
];
const EXPECTED_ROI_ORDER: SelectivityAtlasRoi[] = ['ffa', 'eba', 'ppa'];
const EXPECTED_ROIS: Record<SelectivityAtlasRoi, { offset: number; count: number }> = {
  ffa: { offset: 0, count: 156 },
  eba: { offset: 156, count: 518 },
  ppa: { offset: 674, count: 170 },
};
const EXPECTED_REFERENCE_VERSION = 'ed023e59bd270e210bb8350d348eaaf223d3992214adeb2163a496cc808cefa7';
const EXPECTED_IMAGE_IDS_SHA256 = '1057a45119e774b1bd9c3462664d2d8003ca8d6a1b42c1691f084ce10533a7ca';
const EXPECTED_POINT_COLUMNS = [
  'voxel_index',
  'roi',
  'roi_voxel_index',
  'cluster_label',
  'umap_x',
  'umap_y',
  'umap_z',
  'distance_to_centroid',
];

type JsonObject = Record<string, unknown>;
type FetchLike = typeof fetch;

export class SelectivityAtlasAssetError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SelectivityAtlasAssetError';
  }
}

function object(value: unknown, label: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new SelectivityAtlasAssetError(`${label} must be an object.`);
  }
  return value as JsonObject;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new SelectivityAtlasAssetError(`${label} must be an array.`);
  return value;
}

function string(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new SelectivityAtlasAssetError(`${label} must be a non-empty string.`);
  }
  return value;
}

function integer(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new SelectivityAtlasAssetError(`${label} must be an integer.`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new SelectivityAtlasAssetError(`${label} must be finite.`);
  }
  return value;
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new SelectivityAtlasAssetError(`${label} must be boolean.`);
  return value;
}

function exact<T extends string | boolean>(value: unknown, expected: T, label: string): T {
  if (value !== expected) {
    throw new SelectivityAtlasAssetError(`${label} must be ${JSON.stringify(expected)}.`);
  }
  return expected;
}

function sha256(value: unknown, label: string): string {
  const parsed = string(value, label).toLowerCase();
  if (!/^[\da-f]{64}$/.test(parsed)) {
    throw new SelectivityAtlasAssetError(`${label} must be a SHA-256 digest.`);
  }
  return parsed;
}

function roi(value: unknown, label: string): SelectivityAtlasRoi {
  if (value !== 'ffa' && value !== 'eba' && value !== 'ppa') {
    throw new SelectivityAtlasAssetError(`${label} must be ffa, eba, or ppa.`);
  }
  return value;
}

function validateMethod(value: unknown, label: string): SelectivityAtlasMethod {
  const raw = object(value, label);
  const method: SelectivityAtlasMethod = {
    id: string(raw.id, `${label}.id`),
    objectClustered: string(raw.objectClustered, `${label}.objectClustered`) as 'voxels',
    normalization: string(raw.normalization, `${label}.normalization`),
    pcaComponents: integer(raw.pcaComponents, `${label}.pcaComponents`),
    l2Normalization: boolean(raw.l2Normalization, `${label}.l2Normalization`),
    clustering: string(raw.clustering, `${label}.clustering`),
    clusterCount: integer(raw.clusterCount, `${label}.clusterCount`),
    kmeansNInit: integer(raw.kmeansNInit, `${label}.kmeansNInit`),
    kmeansRandomState: integer(raw.kmeansRandomState, `${label}.kmeansRandomState`),
    embedding: string(raw.embedding, `${label}.embedding`),
    embeddingUsedForClustering: boolean(
      raw.embeddingUsedForClustering,
      `${label}.embeddingUsedForClustering`,
    ),
  };
  if (
    method.id !== 'mentor-selectivity-k14-v1'
    || method.objectClustered !== 'voxels'
    || method.normalization !== 'per-voxel z-score across images (ddof=0, epsilon=1e-9)'
    || method.pcaComponents !== 35
    || !method.l2Normalization
    || method.clustering !== 'K-means in L2-normalized PCA selectivity space'
    || method.clusterCount !== 14
    || method.kmeansNInit !== 25
    || method.kmeansRandomState !== 0
    || method.embedding !== '3D UMAP (n_neighbors=25, min_dist=0.1, metric=cosine, random_state=42); display only'
    || method.embeddingUsedForClustering
  ) {
    throw new SelectivityAtlasAssetError(`${label} does not match the reviewed mentor method.`);
  }
  return method;
}

function validateModelEntry(value: unknown, label: string, model: string): SelectivityAtlasManifestModel {
  const raw = object(value, label);
  const pointCount = integer(raw.pointCount, `${label}.pointCount`);
  if (pointCount !== 844) throw new SelectivityAtlasAssetError(`${label}.pointCount must be 844.`);
  const entry = {
    pointsUrl: string(raw.pointsUrl, `${label}.pointsUrl`),
    summaryUrl: string(raw.summaryUrl, `${label}.summaryUrl`),
    pointsSha256: sha256(raw.pointsSha256, `${label}.pointsSha256`),
    summarySha256: sha256(raw.summarySha256, `${label}.summarySha256`),
    pointCount,
  };
  if (entry.pointsUrl !== `${model}/points.csv` || entry.summaryUrl !== `${model}/summary.json`) {
    throw new SelectivityAtlasAssetError(`${label} contains an unexpected asset URL.`);
  }
  return entry;
}

export function validateSelectivityAtlasManifest(value: unknown): SelectivityAtlasManifest {
  const raw = object(value, 'selectivity-atlas manifest');
  if (raw.schemaVersion !== 1) {
    throw new SelectivityAtlasAssetError('selectivity-atlas manifest.schemaVersion must be 1.');
  }
  const imageCount = integer(raw.imageCount, 'selectivity-atlas manifest.imageCount');
  const voxelCount = integer(raw.voxelCount, 'selectivity-atlas manifest.voxelCount');
  const clusterCount = integer(raw.clusterCount, 'selectivity-atlas manifest.clusterCount');
  if (imageCount !== 185 || voxelCount !== 844 || clusterCount !== 14) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas manifest dimensions are incompatible.');
  }
  const rawRoiOrder = array(raw.roiOrder, 'selectivity-atlas manifest.roiOrder');
  const roiOrder = rawRoiOrder.map((entry, index) => roi(entry, `selectivity-atlas manifest.roiOrder[${index}]`));
  if (roiOrder.some((entry, index) => entry !== EXPECTED_ROI_ORDER[index]) || roiOrder.length !== 3) {
    throw new SelectivityAtlasAssetError('selectivity-atlas manifest.roiOrder must be ffa, eba, ppa.');
  }
  const rawRois = object(raw.rois, 'selectivity-atlas manifest.rois');
  const rois = Object.fromEntries(EXPECTED_ROI_ORDER.map((name) => {
    const entry = object(rawRois[name], `selectivity-atlas manifest.rois.${name}`);
    const parsed = {
      offset: integer(entry.offset, `selectivity-atlas manifest.rois.${name}.offset`),
      count: integer(entry.count, `selectivity-atlas manifest.rois.${name}.count`),
    };
    if (parsed.offset !== EXPECTED_ROIS[name].offset || parsed.count !== EXPECTED_ROIS[name].count) {
      throw new SelectivityAtlasAssetError(`selectivity-atlas manifest.rois.${name} is incompatible.`);
    }
    return [name, parsed];
  })) as SelectivityAtlasManifest['rois'];
  const rawModels = object(raw.models, 'selectivity-atlas manifest.models');
  const models = Object.fromEntries(Object.entries(rawModels).map(([model, entry]) => [
    model,
    validateModelEntry(entry, `selectivity-atlas manifest.models.${model}`, model),
  ]));
  const supportedModels = array(raw.supportedModels, 'selectivity-atlas manifest.supportedModels')
    .map((entry, index) => string(entry, `selectivity-atlas manifest.supportedModels[${index}]`));
  if (
    supportedModels.length !== 26
    || supportedModels.length !== EXPECTED_MODELS.length
    || raw.subject !== 's1'
    || supportedModels.some((model) => !/^[a-z0-9_]+$/.test(model))
    || supportedModels.some((model) => !EXPECTED_MODELS.includes(model))
    || new Set(supportedModels).size !== supportedModels.length
    || supportedModels.some((model) => !models[model])
    || Object.keys(models).some((model) => !supportedModels.includes(model))
  ) {
    throw new SelectivityAtlasAssetError('selectivity-atlas subject/model coverage is incompatible.');
  }
  const referenceVersion = sha256(raw.referenceVersion, 'selectivity-atlas manifest.referenceVersion');
  const imageIdsSha256 = sha256(raw.imageIdsSha256, 'selectivity-atlas manifest.imageIdsSha256');
  if (referenceVersion !== EXPECTED_REFERENCE_VERSION || imageIdsSha256 !== EXPECTED_IMAGE_IDS_SHA256) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas reference identity is incompatible.');
  }
  return {
    schemaVersion: 1,
    artifact: exact(raw.artifact, 'voxel-selectivity-atlas', 'selectivity-atlas manifest.artifact'),
    dataset: exact(raw.dataset, 'nsd_1000', 'selectivity-atlas manifest.dataset'),
    stimulusSet: exact(raw.stimulusSet, 'murty185', 'selectivity-atlas manifest.stimulusSet'),
    subject: exact(raw.subject, 's1', 'selectivity-atlas manifest.subject'),
    responseSpace: exact(
      raw.responseSpace,
      'nsd_1000_subject1_floc_t7_v1',
      'selectivity-atlas manifest.responseSpace',
    ),
    responseKind: exact(
      raw.responseKind,
      'model_predictions',
      'selectivity-atlas manifest.responseKind',
    ),
    embeddingIsAnatomical: exact(
      raw.embeddingIsAnatomical,
      false,
      'selectivity-atlas manifest.embeddingIsAnatomical',
    ),
    imageCount,
    voxelCount,
    clusterCount,
    roiOrder,
    rois,
    method: validateMethod(raw.method, 'selectivity-atlas manifest.method'),
    k14StabilityCaveat: string(
      raw.k14StabilityCaveat,
      'selectivity-atlas manifest.k14StabilityCaveat',
    ),
    supportedModels,
    referenceVersion,
    imageIdsSha256,
    models,
  };
}

function validateStimuli(value: unknown, label: string): SelectivityAtlasStimulus[] {
  const stimuli = array(value, label).map((entry, index) => {
    const raw = object(entry, `${label}[${index}]`);
    const imageId = string(raw.imageId, `${label}[${index}].imageId`);
    const imageIndex = Number(imageId.slice(6, 9));
    if (!/^image_\d{3}\.png$/.test(imageId) || imageIndex < 1 || imageIndex > 185) {
      throw new SelectivityAtlasAssetError(`${label}[${index}].imageId is not canonical.`);
    }
    return { imageId, meanZ: finite(raw.meanZ, `${label}[${index}].meanZ`) };
  });
  if (stimuli.length !== 5 || new Set(stimuli.map((stimulus) => stimulus.imageId)).size !== stimuli.length) {
    throw new SelectivityAtlasAssetError(`${label} must contain five unique images.`);
  }
  return stimuli;
}

function validateCluster(value: unknown, label: string): SelectivityAtlasCluster {
  const raw = object(value, label);
  const rawCounts = object(raw.roiCounts, `${label}.roiCounts`);
  const roiCounts = Object.fromEntries(EXPECTED_ROI_ORDER.map((name) => [
    name,
    integer(rawCounts[name], `${label}.roiCounts.${name}`),
  ])) as Record<SelectivityAtlasRoi, number>;
  const size = integer(raw.size, `${label}.size`);
  if (
    size <= 0
    || Object.values(roiCounts).some((count) => count < 0)
    || Object.values(roiCounts).reduce((sum, count) => sum + count, 0) !== size
  ) {
    throw new SelectivityAtlasAssetError(`${label}.roiCounts do not sum to size.`);
  }
  const meanSilhouette = raw.meanSilhouette === null
    ? null
    : finite(raw.meanSilhouette, `${label}.meanSilhouette`);
  if (meanSilhouette !== null && (meanSilhouette < -1 || meanSilhouette > 1)) {
    throw new SelectivityAtlasAssetError(`${label}.meanSilhouette must be between -1 and 1.`);
  }
  return {
    label: integer(raw.label, `${label}.label`),
    size,
    roiCounts,
    meanSilhouette,
    preferredImages: validateStimuli(raw.preferredImages, `${label}.preferredImages`),
    suppressedImages: validateStimuli(raw.suppressedImages, `${label}.suppressedImages`),
  };
}

export function validateSelectivityAtlasSummary(
  value: unknown,
  manifest: SelectivityAtlasManifest,
  model: string,
): SelectivityAtlasSummary {
  const raw = object(value, 'selectivity-atlas summary');
  if (raw.schemaVersion !== 1 || raw.model !== model) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas summary identity does not match the request.');
  }
  const imageCount = integer(raw.imageCount, 'selectivity-atlas summary.imageCount');
  const voxelCount = integer(raw.voxelCount, 'selectivity-atlas summary.voxelCount');
  const clusterCount = integer(raw.clusterCount, 'selectivity-atlas summary.clusterCount');
  if (
    imageCount !== manifest.imageCount
    || voxelCount !== manifest.voxelCount
    || clusterCount !== manifest.clusterCount
    || raw.dataset !== manifest.dataset
    || raw.stimulusSet !== manifest.stimulusSet
    || raw.subject !== manifest.subject
    || raw.responseSpace !== manifest.responseSpace
    || raw.responseKind !== manifest.responseKind
    || raw.embeddingIsAnatomical !== false
    || raw.k14StabilityCaveat !== manifest.k14StabilityCaveat
  ) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas summary does not match its manifest.');
  }
  const clusters = array(raw.clusters, 'selectivity-atlas summary.clusters')
    .map((entry, index) => validateCluster(entry, `selectivity-atlas summary.clusters[${index}]`));
  const labels = clusters.map((cluster) => cluster.label).sort((a, b) => a - b);
  if (
    clusters.length !== clusterCount
    || labels.some((label, index) => label !== index)
    || clusters.reduce((sum, cluster) => sum + cluster.size, 0) !== voxelCount
  ) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas cluster summaries are incomplete.');
  }
  const sourceFiles = array(raw.sourceFiles, 'selectivity-atlas summary.sourceFiles').map((entry, index) => {
    const source = object(entry, `selectivity-atlas summary.sourceFiles[${index}]`);
    const sourceRoi = roi(source.roi, `selectivity-atlas summary.sourceFiles[${index}].roi`);
    const fileName = string(source.fileName, `selectivity-atlas summary.sourceFiles[${index}].fileName`);
    if (!fileName.startsWith(`murtylab_${model}_nsd_1000_${sourceRoi}_`) || !fileName.endsWith('.csv')) {
      throw new SelectivityAtlasAssetError(`selectivity-atlas source file ${fileName} is incompatible.`);
    }
    return {
      roi: sourceRoi,
      fileName,
      sha256: sha256(source.sha256, `selectivity-atlas summary.sourceFiles[${index}].sha256`),
    };
  });
  if (
    sourceFiles.length !== EXPECTED_ROI_ORDER.length
    || sourceFiles.some((source, index) => source.roi !== EXPECTED_ROI_ORDER[index])
  ) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas source file order is incompatible.');
  }
  const rawStability = object(raw.splitHalfStability, 'selectivity-atlas summary.splitHalfStability');
  const iterations = integer(rawStability.iterations, 'selectivity-atlas summary.splitHalfStability.iterations');
  const adjustedRandIndices = array(
    rawStability.adjustedRandIndices,
    'selectivity-atlas summary.splitHalfStability.adjustedRandIndices',
  ).map((entry, index) => finite(
    entry,
    `selectivity-atlas summary.splitHalfStability.adjustedRandIndices[${index}]`,
  ));
  const meanAdjustedRandIndex = finite(
    rawStability.meanAdjustedRandIndex,
    'selectivity-atlas summary.splitHalfStability.meanAdjustedRandIndex',
  );
  const minAdjustedRandIndex = finite(
    rawStability.minAdjustedRandIndex,
    'selectivity-atlas summary.splitHalfStability.minAdjustedRandIndex',
  );
  const threshold = finite(rawStability.threshold, 'selectivity-atlas summary.splitHalfStability.threshold');
  const randomSeed = integer(
    rawStability.randomSeed,
    'selectivity-atlas summary.splitHalfStability.randomSeed',
  );
  const meanPassesThreshold = boolean(
    rawStability.meanPassesThreshold,
    'selectivity-atlas summary.splitHalfStability.meanPassesThreshold',
  );
  const allPassThreshold = boolean(
    rawStability.allPassThreshold,
    'selectivity-atlas summary.splitHalfStability.allPassThreshold',
  );
  if (
    iterations !== 6
    || randomSeed !== 42
    || adjustedRandIndices.length !== iterations
    || adjustedRandIndices.some((score) => score < -1 || score > 1)
    || threshold !== 0.7
    || meanPassesThreshold !== (meanAdjustedRandIndex >= threshold)
    || allPassThreshold !== adjustedRandIndices.every((score) => score >= threshold)
    || Math.abs(Math.min(...adjustedRandIndices) - minAdjustedRandIndex) > 1e-12
    || Math.abs(
      adjustedRandIndices.reduce((sum, score) => sum + score, 0) / iterations - meanAdjustedRandIndex
    ) > 1e-12
  ) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas split-half stability summary is inconsistent.');
  }
  const silhouette = finite(raw.silhouette, 'selectivity-atlas summary.silhouette');
  const pcaExplainedVarianceRatioSum = finite(
    raw.pcaExplainedVarianceRatioSum,
    'selectivity-atlas summary.pcaExplainedVarianceRatioSum',
  );
  if (silhouette < -1 || silhouette > 1 || pcaExplainedVarianceRatioSum <= 0 || pcaExplainedVarianceRatioSum > 1) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas fit metrics are out of range.');
  }
  return {
    schemaVersion: 1,
    model,
    dataset: manifest.dataset,
    stimulusSet: manifest.stimulusSet,
    subject: manifest.subject,
    responseSpace: manifest.responseSpace,
    responseKind: manifest.responseKind,
    embeddingIsAnatomical: false,
    imageCount,
    voxelCount,
    clusterCount,
    silhouette,
    pcaExplainedVarianceRatioSum,
    splitHalfStability: {
      iterations,
      randomSeed,
      adjustedRandIndices,
      meanAdjustedRandIndex,
      minAdjustedRandIndex,
      threshold,
      meanPassesThreshold,
      allPassThreshold,
    },
    method: validateMethod(raw.method, 'selectivity-atlas summary.method'),
    k14StabilityCaveat: string(
      raw.k14StabilityCaveat,
      'selectivity-atlas summary.k14StabilityCaveat',
    ),
    sourceFiles,
    clusters,
  };
}

function parseCsvInteger(value: string | undefined, label: string): number {
  if (value === undefined || value.trim() === '') throw new SelectivityAtlasAssetError(`${label} is missing.`);
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new SelectivityAtlasAssetError(`${label} must be an integer.`);
  return parsed;
}

function parseCsvFinite(value: string | undefined, label: string): number {
  if (value === undefined || value.trim() === '') throw new SelectivityAtlasAssetError(`${label} is missing.`);
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new SelectivityAtlasAssetError(`${label} must be finite.`);
  return parsed;
}

export function parseSelectivityAtlasPoints(
  text: string,
  manifest: SelectivityAtlasManifest,
  summary: SelectivityAtlasSummary,
): SelectivityAtlasPoint[] {
  const rows = csvParse(text);
  if (
    rows.columns.length !== EXPECTED_POINT_COLUMNS.length
    || rows.columns.some((column, index) => column !== EXPECTED_POINT_COLUMNS[index])
  ) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas points CSV schema is incompatible.');
  }
  if (rows.length !== manifest.voxelCount) {
    throw new SelectivityAtlasAssetError(`The selectivity-atlas points CSV must contain ${manifest.voxelCount} rows.`);
  }
  const points = rows.map((raw, index): SelectivityAtlasPoint => {
    const pointRoi = roi(raw.roi, `selectivity-atlas point ${index}.roi`);
    const point: SelectivityAtlasPoint = {
      voxelIndex: parseCsvInteger(raw.voxel_index, `selectivity-atlas point ${index}.voxel_index`),
      roi: pointRoi,
      roiVoxelIndex: parseCsvInteger(raw.roi_voxel_index, `selectivity-atlas point ${index}.roi_voxel_index`),
      clusterLabel: parseCsvInteger(raw.cluster_label, `selectivity-atlas point ${index}.cluster_label`),
      umapX: parseCsvFinite(raw.umap_x, `selectivity-atlas point ${index}.umap_x`),
      umapY: parseCsvFinite(raw.umap_y, `selectivity-atlas point ${index}.umap_y`),
      umapZ: parseCsvFinite(raw.umap_z, `selectivity-atlas point ${index}.umap_z`),
      distanceToCentroid: parseCsvFinite(
        raw.distance_to_centroid,
        `selectivity-atlas point ${index}.distance_to_centroid`,
      ),
    };
    if (
      point.voxelIndex !== index
      || point.voxelIndex !== manifest.rois[pointRoi].offset + point.roiVoxelIndex
      || point.roiVoxelIndex < 0
      || point.roiVoxelIndex >= manifest.rois[pointRoi].count
      || point.clusterLabel < 0
      || point.clusterLabel >= manifest.clusterCount
      || point.distanceToCentroid < 0
    ) {
      throw new SelectivityAtlasAssetError(`selectivity-atlas point ${index} is out of range.`);
    }
    return point;
  });
  const pointKeys = new Set(points.map((point) => `${point.roi}:${point.roiVoxelIndex}`));
  if (pointKeys.size !== points.length) {
    throw new SelectivityAtlasAssetError('The selectivity-atlas points contain duplicate ROI voxel indices.');
  }
  for (const cluster of summary.clusters) {
    const members = points.filter((point) => point.clusterLabel === cluster.label);
    if (members.length !== cluster.size) {
      throw new SelectivityAtlasAssetError(`Cluster ${cluster.label} size does not match its points.`);
    }
    for (const name of EXPECTED_ROI_ORDER) {
      if (members.filter((point) => point.roi === name).length !== cluster.roiCounts[name]) {
        throw new SelectivityAtlasAssetError(`Cluster ${cluster.label} ${name.toUpperCase()} count is inconsistent.`);
      }
    }
  }
  return points;
}

function resolveAssetUrl(path: string, manifestUrl: string): string {
  const base = typeof window === 'undefined' ? 'http://localhost/' : window.location.href;
  return new URL(path, new URL(manifestUrl, base)).toString();
}

async function fetchText(url: string, fetcher: FetchLike, signal?: AbortSignal): Promise<string> {
  const response = await fetcher(url, { signal });
  if (!response.ok) throw new SelectivityAtlasAssetError(`Could not load ${url} (${response.status}).`);
  return response.text();
}

async function verifySha256(text: string, expected: string, label: string): Promise<void> {
  if (!globalThis.crypto?.subtle) {
    throw new SelectivityAtlasAssetError(`Cannot verify ${label}: Web Crypto is unavailable.`);
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  const actual = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
  if (actual !== expected) throw new SelectivityAtlasAssetError(`${label} failed its SHA-256 check.`);
}

/** Verify the reviewed artifact manifest before trusting its model assets or availability. */
export async function loadSelectivityAtlasManifest(
  fetcher: FetchLike = fetch,
  signal?: AbortSignal,
): Promise<SelectivityAtlasManifest> {
  if (signal?.aborted) throw new DOMException('The request was aborted.', 'AbortError');
  const manifestText = await fetchText(SELECTIVITY_ATLAS_MANIFEST_PATH, fetcher, signal);
  await verifySha256(manifestText, MANIFEST_SHA256, 'Selectivity atlas manifest');
  let rawManifest: unknown;
  try {
    rawManifest = JSON.parse(manifestText);
  } catch {
    throw new SelectivityAtlasAssetError('The selectivity-atlas manifest is not valid JSON.');
  }
  return validateSelectivityAtlasManifest(rawManifest);
}

export async function loadSelectivityAtlasAssets(
  model: string,
  fetcher: FetchLike = fetch,
  signal?: AbortSignal,
): Promise<SelectivityAtlasAssets> {
  const manifest = await loadSelectivityAtlasManifest(fetcher, signal);
  const entry = manifest.models[model];
  if (!entry) throw new SelectivityAtlasAssetError(`No Subject 1 selectivity atlas is available for ${model}.`);
  const summaryUrl = resolveAssetUrl(entry.summaryUrl, SELECTIVITY_ATLAS_MANIFEST_PATH);
  const pointsUrl = resolveAssetUrl(entry.pointsUrl, SELECTIVITY_ATLAS_MANIFEST_PATH);
  const [summaryText, pointsText] = await Promise.all([
    fetchText(summaryUrl, fetcher, signal),
    fetchText(pointsUrl, fetcher, signal),
  ]);
  await Promise.all([
    verifySha256(summaryText, entry.summarySha256, `${model} selectivity summary`),
    verifySha256(pointsText, entry.pointsSha256, `${model} selectivity points`),
  ]);
  let rawSummary: unknown;
  try {
    rawSummary = JSON.parse(summaryText);
  } catch {
    throw new SelectivityAtlasAssetError('The selectivity-atlas summary is not valid JSON.');
  }
  const summary = validateSelectivityAtlasSummary(rawSummary, manifest, model);
  const points = parseSelectivityAtlasPoints(pointsText, manifest, summary);
  return { manifest, summary, points };
}
