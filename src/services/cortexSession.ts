import {
  CANONICAL_ROIS,
  type CanonicalRoi,
  type CortexComparisonSession,
  type FeatureHierarchicalAssignment,
  type FeatureProjection,
  type FeatureReferenceNeighbor,
  type HierarchicalAssignment,
  type OptionalProjectionResults,
  type PublicReferenceManifest,
  type ReferenceNeighbor,
  type SessionLoadErrorKind,
  type SessionRoiResult,
  type SessionStimulus,
  type StimulusRoiResult,
  type VoxelProjection,
} from '../types/session';

const DEFAULT_CORTEX_CORE_URL = 'http://localhost:5556';
const DEFAULT_CORTEX_APP_URL = 'http://localhost:5173';
const PUBLIC_REFERENCE_MANIFEST_PATH = '/data/reference_manifest.json';

type JsonObject = Record<string, unknown>;

export class CortexSessionError extends Error {
  readonly kind: SessionLoadErrorKind;

  constructor(kind: SessionLoadErrorKind, message: string) {
    super(message);
    this.name = 'CortexSessionError';
    this.kind = kind;
  }
}

function object(value: unknown, label: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new CortexSessionError('invalid', `${label} must be an object.`);
  }
  return value as JsonObject;
}

function string(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new CortexSessionError('invalid', `${label} must be a non-empty string.`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new CortexSessionError('invalid', `${label} must be a finite number.`);
  }
  return value;
}

function integer(value: unknown, label: string): number {
  const parsed = finite(value, label);
  if (!Number.isInteger(parsed)) {
    throw new CortexSessionError('invalid', `${label} must be an integer.`);
  }
  return parsed;
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {
    throw new CortexSessionError('invalid', `${label} must be a boolean.`);
  }
  return value;
}

function identifier(value: unknown, label: string): string {
  if (typeof value === 'string' && value.trim() !== '') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  throw new CortexSessionError('invalid', `${label} must be a non-empty identifier.`);
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new CortexSessionError('invalid', `${label} must be an array.`);
  }
  return value;
}

function nullableFinite(value: unknown, label: string): number | null {
  return value === null || value === undefined ? null : finite(value, label);
}

function coreBaseUrl(): string {
  const configured = import.meta.env.VITE_CORTEX_CORE_URL?.trim();
  return (configured || DEFAULT_CORTEX_CORE_URL).replace(/\/+$/, '');
}

export function getCortexAppUrl(): string {
  const configured = import.meta.env.VITE_CORTEX_APP_URL?.trim();
  return configured || DEFAULT_CORTEX_APP_URL;
}

export function resolveCortexUrl(
  pathOrUrl: string,
  baseUrl = coreBaseUrl(),
  browserOrigin = typeof window === 'undefined' ? 'http://localhost:5174' : window.location.origin,
): string {
  try {
    const normalizedBase = baseUrl.replace(/\/+$/, '');
    const base = new URL(
      `${normalizedBase.startsWith('/') ? normalizedBase : `/${normalizedBase}`}/`,
      browserOrigin,
    );
    const isAbsoluteBase = /^[a-z][a-z\d+.-]*:\/\//i.test(normalizedBase);
    const effectiveBase = isAbsoluteBase ? new URL(`${normalizedBase}/`) : base;
    const resolvedPath = pathOrUrl.startsWith('/') && effectiveBase.pathname !== '/'
      ? `${effectiveBase.pathname.replace(/\/+$/, '')}${pathOrUrl}`
      : pathOrUrl;
    const resolved = new URL(resolvedPath, effectiveBase);
    if (resolved.origin !== effectiveBase.origin) {
      throw new Error('cross-origin session asset');
    }
    return resolved.toString();
  } catch {
    throw new CortexSessionError('invalid', 'A session image URL was invalid.');
  }
}

export function extractSessionToken(hash: string): string | null {
  const params = new URLSearchParams(hash.startsWith('#') ? hash.slice(1) : hash);
  const token = params.get('session')?.trim() ?? '';
  if (!token) return null;
  if (!/^[A-Za-z0-9_-]{16,256}$/.test(token)) {
    throw new CortexSessionError('invalid', 'The visualization session token is malformed.');
  }
  return token;
}

export function clearSessionFragment(locationValue = window.location, historyValue = window.history): void {
  historyValue.replaceState(null, '', `${locationValue.pathname}${locationValue.search}`);
}

function parseNeighbor(value: unknown, label: string, baseUrl: string): ReferenceNeighbor {
  const raw = object(value, label);
  const imageName = string(raw.image_name, `${label}.image_name`);
  // Reference images are fixed InfoViz assets. Core supplies the path for schema
  // completeness but does not serve this collection.
  string(raw.image_url, `${label}.image_url`);
  const distance = raw.distance === undefined || raw.distance === null
    ? undefined
    : finite(raw.distance, `${label}.distance`);
  const rank = finite(raw.rank, `${label}.rank`);
  if (rank < 1 || rank > 185) {
    throw new CortexSessionError('invalid', `${label}.rank must be between 1 and 185.`);
  }
  return {
    imageName,
    imageUrl: `/images/${encodeURIComponent(imageName)}`,
    value: finite(raw.value, `${label}.value`),
    rank,
    ...(distance === undefined ? {} : { distance }),
  };
}

function parseNeighborList(value: unknown, label: string, baseUrl: string): ReferenceNeighbor[] {
  return array(value, label).map((neighbor, index) => parseNeighbor(neighbor, `${label}[${index}]`, baseUrl));
}

function parseProjection(value: unknown, label: string, baseUrl: string): VoxelProjection {
  const raw = object(value, label);
  return {
    x: finite(raw.x, `${label}.x`),
    y: finite(raw.y, `${label}.y`),
    clusterId: identifier(raw.cluster_id, `${label}.cluster_id`),
    centroidDistance: finite(raw.centroid_distance, `${label}.centroid_distance`),
    nearestReferenceMembers: parseNeighborList(
      raw.nearest_reference_members,
      `${label}.nearest_reference_members`,
      baseUrl,
    ),
  };
}

function parseHierarchicalAssignment(value: unknown, label: string, baseUrl: string): HierarchicalAssignment {
  const raw = object(value, label);
  if (raw.method !== 'nearest-medoid assignment') {
    throw new CortexSessionError('invalid', `${label}.method must be "nearest-medoid assignment".`);
  }
  return {
    method: 'nearest-medoid assignment',
    clusterId: identifier(raw.cluster_id, `${label}.cluster_id`),
    medoid: parseNeighbor(raw.medoid, `${label}.medoid`, baseUrl),
    distance: finite(raw.distance, `${label}.distance`),
  };
}

function parseStimulusRoiResult(value: unknown, label: string, baseUrl: string): StimulusRoiResult {
  const raw = object(value, label);
  const referenceRank = finite(raw.reference_rank, `${label}.reference_rank`);
  const referencePercentile = finite(raw.reference_percentile, `${label}.reference_percentile`);
  const sessionRank = integer(raw.session_rank, `${label}.session_rank`);
  const sessionCount = integer(raw.session_count, `${label}.session_count`);
  if (referenceRank < 1 || referenceRank > 186 || referencePercentile < 0 || referencePercentile > 100) {
    throw new CortexSessionError('invalid', `${label} contains an out-of-range reference rank or percentile.`);
  }
  if (sessionRank < 1 || sessionRank > sessionCount) {
    throw new CortexSessionError('invalid', `${label} contains an out-of-range session rank.`);
  }
  return {
    stimulusId: string(raw.stimulus_id, `${label}.stimulus_id`),
    mean: finite(raw.mean, `${label}.mean`),
    sem: nullableFinite(raw.sem, `${label}.sem`),
    referenceRank,
    referencePercentile,
    sessionRank,
    sessionCount,
    predecessor: raw.predecessor === null || raw.predecessor === undefined
      ? null
      : parseNeighbor(raw.predecessor, `${label}.predecessor`, baseUrl),
    successor: raw.successor === null || raw.successor === undefined
      ? null
      : parseNeighbor(raw.successor, `${label}.successor`, baseUrl),
    closestByMean: parseNeighborList(raw.closest_by_mean, `${label}.closest_by_mean`, baseUrl),
    closestInVoxelSpace: parseNeighborList(
      raw.closest_in_voxel_space,
      `${label}.closest_in_voxel_space`,
      baseUrl,
    ),
    entersTop: boolean(raw.enters_top, `${label}.enters_top`),
    entersBottom: boolean(raw.enters_bottom, `${label}.enters_bottom`),
    voxelProjection: parseProjection(raw.voxel_projection, `${label}.voxel_projection`, baseUrl),
    hierarchicalAssignment: parseHierarchicalAssignment(
      raw.hierarchical_assignment,
      `${label}.hierarchical_assignment`,
      baseUrl,
    ),
  };
}

function parseRoiResult(value: unknown, roi: CanonicalRoi, baseUrl: string): SessionRoiResult {
  const raw = object(value, `roi_results.${roi}`);
  return {
    stimuli: array(raw.stimuli, `roi_results.${roi}.stimuli`).map((stimulus, index) =>
      parseStimulusRoiResult(stimulus, `roi_results.${roi}.stimuli[${index}]`, baseUrl)),
    topReferences: parseNeighborList(raw.top_references, `roi_results.${roi}.top_references`, baseUrl),
    bottomReferences: parseNeighborList(raw.bottom_references, `roi_results.${roi}.bottom_references`, baseUrl),
  };
}

function parseFeatureNeighbor(
  value: unknown,
  label: string,
  fallbackDistance?: number,
): FeatureReferenceNeighbor {
  if (typeof value === 'string' && value.trim() !== '') {
    return {
      imageName: value,
      imageUrl: `/images/${encodeURIComponent(value)}`,
      distance: fallbackDistance ?? 0,
    };
  }
  const raw = object(value, label);
  const imageName = string(raw.image_name, `${label}.image_name`);
  if (raw.image_url !== undefined && raw.image_url !== null) string(raw.image_url, `${label}.image_url`);
  return {
    imageName,
    imageUrl: `/images/${encodeURIComponent(imageName)}`,
    distance: raw.distance === undefined && fallbackDistance !== undefined
      ? fallbackDistance
      : finite(raw.distance, `${label}.distance`),
  };
}

function parseFeatureProjection(value: unknown, label: string): FeatureProjection {
  const raw = object(value, label);
  return {
    x: finite(raw.x, `${label}.x`),
    y: finite(raw.y, `${label}.y`),
    clusterId: identifier(raw.cluster_id, `${label}.cluster_id`),
    centroidDistance: finite(raw.centroid_distance, `${label}.centroid_distance`),
    nearestReferenceMembers: array(raw.nearest_reference_members, `${label}.nearest_reference_members`)
      .map((neighbor, index) => parseFeatureNeighbor(neighbor, `${label}.nearest_reference_members[${index}]`)),
  };
}

function parseFeatureHierarchicalAssignment(value: unknown, label: string): FeatureHierarchicalAssignment {
  const raw = object(value, label);
  if (raw.method !== 'nearest-medoid assignment') {
    throw new CortexSessionError('invalid', `${label}.method must be "nearest-medoid assignment".`);
  }
  const distance = finite(raw.distance, `${label}.distance`);
  return {
    method: 'nearest-medoid assignment',
    clusterId: identifier(raw.cluster_id, `${label}.cluster_id`),
    medoid: parseFeatureNeighbor(raw.medoid, `${label}.medoid`, distance),
    distance,
  };
}

function parseOptionalProjectionResults(value: unknown, label: string): OptionalProjectionResults {
  const raw = object(value, label);
  return {
    stimuli: array(raw.stimuli, `${label}.stimuli`).map((entry, index) => {
      const itemLabel = `${label}.stimuli[${index}]`;
      const item = object(entry, itemLabel);
      const projectionValue = item.projection ?? item.voxel_projection ?? item;
      return {
        stimulusId: string(item.stimulus_id, `${itemLabel}.stimulus_id`),
        projection: projectionValue === null || projectionValue === undefined
          ? null
          : parseFeatureProjection(projectionValue, `${itemLabel}.projection`),
        hierarchicalAssignment: item.hierarchical_assignment === null || item.hierarchical_assignment === undefined
          ? null
          : parseFeatureHierarchicalAssignment(item.hierarchical_assignment, `${itemLabel}.hierarchical_assignment`),
      };
    }),
  };
}

function parseStimulus(value: unknown, index: number, baseUrl: string): SessionStimulus {
  const label = `stimuli[${index}]`;
  const raw = object(value, label);
  const color = string(raw.color, `${label}.color`);
  if (color.length > 40 || /[;{}]|url\s*\(/i.test(color)) {
    throw new CortexSessionError('invalid', `${label}.color is invalid.`);
  }
  return {
    id: string(raw.id, `${label}.id`),
    displayName: string(raw.display_name, `${label}.display_name`),
    imageUrl: resolveCortexUrl(string(raw.image_url, `${label}.image_url`), baseUrl),
    color,
  };
}

export function validateComparisonSession(
  value: unknown,
  baseUrl = coreBaseUrl(),
  now = new Date(),
): CortexComparisonSession {
  const raw = object(value, 'session');
  if (raw.schema_version !== 1) {
    throw new CortexSessionError('invalid', 'This visualization session uses an unsupported schema version.');
  }

  const source = object(raw.source_request, 'source_request');
  const comparison = object(raw.comparison, 'comparison');
  const rawRois = array(comparison.rois, 'comparison.rois').map((roi) => String(roi));
  if (
    rawRois.length !== CANONICAL_ROIS.length
    || CANONICAL_ROIS.some((roi) => !rawRois.includes(roi))
    || new Set(rawRois).size !== CANONICAL_ROIS.length
  ) {
    throw new CortexSessionError('invalid', 'The comparison session must contain exactly FFA, PPA, and EBA.');
  }
  if (comparison.dataset !== 'nsd_1000' || comparison.reference_count !== 185) {
    throw new CortexSessionError('invalid', 'The comparison session is not an NSD-1000 / 185-image reference session.');
  }

  const createdAt = string(raw.created_at, 'created_at');
  const expiresAt = string(raw.expires_at, 'expires_at');
  if (!Number.isFinite(Date.parse(createdAt)) || !Number.isFinite(Date.parse(expiresAt))) {
    throw new CortexSessionError('invalid', 'The session timestamps are invalid.');
  }
  if (Date.parse(expiresAt) <= now.getTime()) {
    throw new CortexSessionError('expired', 'This visualization session has expired. Return to Cortex to prepare a new one.');
  }

  const stimuli = array(raw.stimuli, 'stimuli').map((stimulus, index) => parseStimulus(stimulus, index, baseUrl));
  if (stimuli.length === 0 || new Set(stimuli.map((stimulus) => stimulus.id)).size !== stimuli.length) {
    throw new CortexSessionError('invalid', 'The session must contain at least one uniquely identified stimulus.');
  }

  const rawRoiResults = object(raw.roi_results, 'roi_results');
  const roiResults = Object.fromEntries(
    CANONICAL_ROIS.map((roi) => [roi, parseRoiResult(rawRoiResults[roi], roi, baseUrl)]),
  ) as Record<CanonicalRoi, SessionRoiResult>;
  const expectedIds = [...stimuli.map((stimulus) => stimulus.id)].sort();
  for (const roi of CANONICAL_ROIS) {
    const actualIds = roiResults[roi].stimuli.map((stimulus) => stimulus.stimulusId).sort();
    if (actualIds.length !== expectedIds.length || actualIds.some((id, index) => id !== expectedIds[index])) {
      throw new CortexSessionError('invalid', `ROI ${roi.toUpperCase()} does not contain every uploaded stimulus exactly once.`);
    }
    const expectedSessionCount = 185 + stimuli.length;
    if (roiResults[roi].stimuli.some((stimulus) => stimulus.sessionCount !== expectedSessionCount)) {
      throw new CortexSessionError('invalid', `ROI ${roi.toUpperCase()} reports an inconsistent session image count.`);
    }
    const sessionRanks = roiResults[roi].stimuli.map((stimulus) => stimulus.sessionRank);
    if (new Set(sessionRanks).size !== sessionRanks.length) {
      throw new CortexSessionError('invalid', `ROI ${roi.toUpperCase()} reports duplicate combined ranks for uploads.`);
    }
  }

  const visualResults = raw.visual_results === undefined || raw.visual_results === null
    ? undefined
    : parseOptionalProjectionResults(raw.visual_results, 'visual_results');
  const dreamsimResults = raw.dreamsim_results === undefined || raw.dreamsim_results === null
    ? undefined
    : parseOptionalProjectionResults(raw.dreamsim_results, 'dreamsim_results');
  for (const [label, results] of [['visual_results', visualResults], ['dreamsim_results', dreamsimResults]] as const) {
    if (!results) continue;
    const actualIds = results.stimuli.map((stimulus) => stimulus.stimulusId).sort();
    if (actualIds.length !== expectedIds.length || actualIds.some((id, index) => id !== expectedIds[index])) {
      throw new CortexSessionError('invalid', `${label} does not contain every uploaded stimulus exactly once.`);
    }
  }

  return {
    schemaVersion: 1,
    createdAt,
    expiresAt,
    sourceRequest: {
      model: string(source.model, 'source_request.model'),
      dataset: string(source.dataset, 'source_request.dataset'),
      roi: string(source.roi, 'source_request.roi'),
    },
    comparison: {
      model: string(comparison.model, 'comparison.model'),
      dataset: 'nsd_1000',
      rois: [...CANONICAL_ROIS],
      referenceVersion: string(comparison.reference_version, 'comparison.reference_version'),
      referenceCount: 185,
    },
    stimuli,
    roiResults,
    ...(visualResults ? { visualResults } : {}),
    ...(dreamsimResults ? { dreamsimResults } : {}),
  };
}

export function validateReferenceManifest(value: unknown): PublicReferenceManifest {
  const raw = object(value, 'reference manifest');
  if (raw.schema_version !== 1) {
    throw new CortexSessionError('invalid', 'The local reference manifest uses an unsupported schema version.');
  }
  const dataset = raw.dataset;
  if (dataset !== 'nsd_1000') {
    throw new CortexSessionError('invalid', 'The local reference manifest is not for NSD-1000.');
  }
  const rois = array(raw.rois, 'reference manifest.rois').map((roi) => String(roi));
  if (CANONICAL_ROIS.some((roi) => !rois.includes(roi))) {
    throw new CortexSessionError('invalid', 'The local reference manifest is missing a canonical ROI.');
  }
  const rawModels = raw.models ?? raw.supported_models;
  const models = Array.isArray(rawModels)
    ? rawModels.map((model) => String(model))
    : Object.keys(object(rawModels, 'reference manifest.models'));
  return {
    schemaVersion: 1,
    referenceVersion: string(raw.reference_version, 'reference manifest.reference_version'),
    dataset: 'nsd_1000',
    referenceCount: integer(raw.reference_count ?? raw.image_count, 'reference manifest.reference_count'),
    models,
    rois: [...CANONICAL_ROIS],
  };
}

export function assertSessionReferenceCompatible(
  session: CortexComparisonSession,
  manifest: PublicReferenceManifest,
): void {
  if (session.comparison.referenceVersion !== manifest.referenceVersion) {
    throw new CortexSessionError(
      'version-mismatch',
      'This Cortex session was created from a different historical reference version. Return to Cortex and prepare it again.',
    );
  }
  if (manifest.referenceCount !== 185 || !manifest.models.includes(session.comparison.model)) {
    throw new CortexSessionError('version-mismatch', 'The local visualization data does not support this comparison session.');
  }
}

async function readJsonResponse(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new CortexSessionError('invalid', 'The server returned malformed session data.');
  }
}

export async function loadCortexComparisonSession(
  token: string,
  fetcher: typeof fetch = fetch,
): Promise<CortexComparisonSession> {
  const baseUrl = coreBaseUrl();
  let response: Response;
  try {
    response = await fetcher(`${baseUrl}/comparison-sessions/${encodeURIComponent(token)}`, {
      headers: { Accept: 'application/json' },
      method: 'GET',
    });
  } catch {
    throw new CortexSessionError('unavailable', 'Cortex Core could not be reached. The standalone visualization remains available without this session.');
  }
  if (response.status === 410) {
    throw new CortexSessionError('expired', 'This visualization session has expired. Return to Cortex to prepare a new one.');
  }
  if (response.status === 404) {
    throw new CortexSessionError('invalid', 'This visualization session does not exist or is no longer available.');
  }
  if (!response.ok) {
    throw new CortexSessionError('unavailable', `Cortex Core could not load the session (HTTP ${response.status}).`);
  }
  return validateComparisonSession(await readJsonResponse(response), baseUrl);
}

export async function loadPublicReferenceManifest(fetcher: typeof fetch = fetch): Promise<PublicReferenceManifest> {
  let response: Response;
  try {
    response = await fetcher(PUBLIC_REFERENCE_MANIFEST_PATH, { headers: { Accept: 'application/json' } });
  } catch {
    throw new CortexSessionError('unavailable', 'The local reference manifest could not be loaded.');
  }
  if (!response.ok) {
    throw new CortexSessionError('unavailable', `The local reference manifest could not be loaded (HTTP ${response.status}).`);
  }
  return validateReferenceManifest(await readJsonResponse(response));
}
