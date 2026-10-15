import {
  CANONICAL_ROIS,
  SUBJECT_ONE_ROI_VOXEL_COUNTS,
  type BrainMapResults,
  type CanonicalRoi,
  type CortexComparisonSession,
  type FeatureHierarchicalAssignment,
  type FeatureProjection,
  type FeatureReferenceNeighbor,
  type HierarchicalAssignment,
  type JointHierarchy,
  type JointHierarchyMember,
  type OptionalProjectionResults,
  type PublicReferenceManifest,
  type ReferenceNeighbor,
  type SessionLoadErrorKind,
  type SessionModelResults,
  type SessionRoiResult,
  type SessionStimulus,
  type StimulusRoiResult,
  type VoxelProjection,
} from '../types/session';
import { publicPath } from '../utils/publicPath';
import { getSessionModels } from '../utils/sessionModels';

const DEFAULT_CORTEX_CORE_URL = '/cortex-core';
const DEFAULT_CORTEX_APP_URL = 'http://localhost:5173';
const PUBLIC_REFERENCE_MANIFEST_PATH = publicPath('/data/reference_manifest.json');
const SESSION_TOKEN_PATTERN = /^[A-Za-z0-9_-]{16,256}$/;
const MULTI_SESSION_PREFIX = 'multi-v1.';

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
  parseSessionTokens(token);
  return token;
}

function parseSessionTokens(token: string): string[] {
  if (SESSION_TOKEN_PATTERN.test(token)) return [token];
  const tokens = token.startsWith(MULTI_SESSION_PREFIX)
    ? token.slice(MULTI_SESSION_PREFIX.length).split('.')
    : [];
  if (tokens.length < 2 || tokens.length > 4
    || new Set(tokens).size !== tokens.length
    || tokens.some((part) => !SESSION_TOKEN_PATTERN.test(part))) {
    throw new CortexSessionError('invalid', 'The Cortex-Compass session token is malformed.');
  }
  return tokens;
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
    imageUrl: publicPath(`/images/${encodeURIComponent(imageName)}`),
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

type JointHierarchyContext = {
  referenceCount: number;
  uploadedIds: string[];
};

function correlationDistance(value: unknown, label: string): number {
  const parsed = finite(value, label);
  if (parsed < 0 || parsed > 2) {
    throw new CortexSessionError('invalid', `${label} must be between 0 and 2.`);
  }
  return parsed;
}

function parseJointHierarchy(
  value: unknown,
  label: string,
  context: JointHierarchyContext,
): JointHierarchy {
  const raw = object(value, label);
  const requiredMetadata = {
    schema_version: 1,
    method: 'hierarchical_correlation',
    distance_metric: 'correlation',
    linkage_method: 'average',
    normalization: 'reference_zscore',
  };
  for (const [key, expected] of Object.entries(requiredMetadata)) {
    if (raw[key] !== expected) {
      throw new CortexSessionError('invalid', `${label}.${key} must be ${JSON.stringify(expected)}.`);
    }
  }

  const referenceCount = integer(raw.reference_count, `${label}.reference_count`);
  const uploadedCount = integer(raw.uploaded_count, `${label}.uploaded_count`);
  if (referenceCount !== context.referenceCount || uploadedCount !== context.uploadedIds.length) {
    throw new CortexSessionError('invalid', `${label} contains inconsistent reference or uploaded image counts.`);
  }
  const memberCount = referenceCount + uploadedCount;
  const clusterCount = integer(raw.cluster_count, `${label}.cluster_count`);
  if (clusterCount < 1 || clusterCount > 5 || clusterCount >= memberCount) {
    throw new CortexSessionError('invalid', `${label}.cluster_count is out of range.`);
  }
  const silhouette = raw.silhouette === null
    ? null
    : finite(raw.silhouette, `${label}.silhouette`);
  if (
    (silhouette !== null && (silhouette < -1 || silhouette > 1))
    || ((clusterCount === 1) !== (silhouette === null))
  ) {
    throw new CortexSessionError('invalid', `${label}.silhouette must be in [-1, 1] for multiple clusters, or null for one cluster.`);
  }
  const members = array(raw.members, `${label}.members`).map((entry, index): JointHierarchyMember => {
    const memberLabel = `${label}.members[${index}]`;
    const member = object(entry, memberLabel);
    const kind = index < referenceCount ? 'reference' : 'uploaded';
    if (member.kind !== kind) {
      throw new CortexSessionError('invalid', `${memberLabel}.kind must be "${kind}"; reference members must precede uploads.`);
    }
    const id = string(member.id, `${memberLabel}.id`);
    if (kind === 'uploaded' && id !== context.uploadedIds[index - referenceCount]) {
      throw new CortexSessionError('invalid', `${memberLabel}.id does not match the uploaded stimulus order.`);
    }
    const clusterId = integer(member.cluster_id, `${memberLabel}.cluster_id`);
    if (clusterId < 0 || clusterId >= clusterCount) {
      throw new CortexSessionError('invalid', `${memberLabel}.cluster_id is out of range.`);
    }
    return {
      kind,
      id,
      clusterId: String(clusterId),
      distanceToMedoid: correlationDistance(member.distance_to_medoid, `${memberLabel}.distance_to_medoid`),
    };
  });
  if (members.length !== memberCount) {
    throw new CortexSessionError('invalid', `${label}.members must contain every reference and uploaded image exactly once.`);
  }
  const referenceIds = members.slice(0, referenceCount).map((member) => member.id);
  if (new Set(referenceIds).size !== referenceCount) {
    throw new CortexSessionError('invalid', `${label}.members contains duplicate reference image IDs.`);
  }
  if (new Set(members.map((member) => member.clusterId)).size !== clusterCount) {
    throw new CortexSessionError('invalid', `${label}.members must contain contiguous, non-empty cluster IDs.`);
  }

  const rawLinkage = array(raw.linkage, `${label}.linkage`);
  if (rawLinkage.length !== memberCount - 1) {
    throw new CortexSessionError('invalid', `${label}.linkage must contain one merge for every non-root tree node.`);
  }
  const descendantCounts = Array.from({ length: memberCount }, () => 1);
  const usedNodes = new Set<number>();
  let previousDistance = 0;
  const linkage = rawLinkage.map((entry, index): [number, number, number, number] => {
    const rowLabel = `${label}.linkage[${index}]`;
    const row = array(entry, rowLabel);
    if (row.length !== 4) {
      throw new CortexSessionError('invalid', `${rowLabel} must contain two child indices, a distance, and a descendant count.`);
    }
    const left = integer(row[0], `${rowLabel}[0]`);
    const right = integer(row[1], `${rowLabel}[1]`);
    const parent = memberCount + index;
    if (
      left < 0 || right < 0 || left >= parent || right >= parent
      || left === right || usedNodes.has(left) || usedNodes.has(right)
    ) {
      throw new CortexSessionError('invalid', `${rowLabel} contains invalid or reused child nodes.`);
    }
    const distance = correlationDistance(row[2], `${rowLabel}[2]`);
    if (distance < previousDistance - 1e-12) {
      throw new CortexSessionError('invalid', `${rowLabel}[2] must preserve monotonic average-linkage distances.`);
    }
    const count = integer(row[3], `${rowLabel}[3]`);
    if (count !== descendantCounts[left] + descendantCounts[right]) {
      throw new CortexSessionError('invalid', `${rowLabel}[3] does not match its descendant count.`);
    }
    usedNodes.add(left);
    usedNodes.add(right);
    descendantCounts.push(count);
    previousDistance = distance;
    return [left, right, distance, count];
  });
  const root = 2 * memberCount - 2;
  if (usedNodes.size !== root || descendantCounts[root] !== memberCount) {
    throw new CortexSessionError('invalid', `${label}.linkage does not form one complete tree.`);
  }

  const medoidIndices = array(raw.medoid_indices, `${label}.medoid_indices`).map((entry, clusterId) => {
    const index = integer(entry, `${label}.medoid_indices[${clusterId}]`);
    if (index < 0 || index >= memberCount || members[index].clusterId !== String(clusterId)) {
      throw new CortexSessionError('invalid', `${label}.medoid_indices[${clusterId}] must identify a member of that cluster.`);
    }
    if (members[index].distanceToMedoid > 1e-12) {
      throw new CortexSessionError('invalid', `${label}.medoid_indices[${clusterId}] must identify a representative with zero self-distance.`);
    }
    return index;
  });
  if (medoidIndices.length !== clusterCount) {
    throw new CortexSessionError('invalid', `${label}.medoid_indices must contain one representative per cluster.`);
  }

  return {
    schemaVersion: 1,
    method: 'hierarchical_correlation',
    distanceMetric: 'correlation',
    linkageMethod: 'average',
    normalization: 'reference_zscore',
    referenceCount,
    uploadedCount,
    clusterCount,
    silhouette,
    members,
    linkage,
    medoidIndices,
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

function parseRoiResult(
  value: unknown,
  roi: CanonicalRoi,
  baseUrl: string,
  jointContext: JointHierarchyContext,
): SessionRoiResult {
  const raw = object(value, `roi_results.${roi}`);
  return {
    stimuli: array(raw.stimuli, `roi_results.${roi}.stimuli`).map((stimulus, index) =>
      parseStimulusRoiResult(stimulus, `roi_results.${roi}.stimuli[${index}]`, baseUrl)),
    topReferences: parseNeighborList(raw.top_references, `roi_results.${roi}.top_references`, baseUrl),
    bottomReferences: parseNeighborList(raw.bottom_references, `roi_results.${roi}.bottom_references`, baseUrl),
    ...(raw.joint_hierarchy === undefined ? {} : {
      jointHierarchy: parseJointHierarchy(raw.joint_hierarchy, `roi_results.${roi}.joint_hierarchy`, jointContext),
    }),
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
      imageUrl: publicPath(`/images/${encodeURIComponent(value)}`),
      distance: fallbackDistance ?? 0,
    };
  }
  const raw = object(value, label);
  const imageName = string(raw.image_name, `${label}.image_name`);
  if (raw.image_url !== undefined && raw.image_url !== null) string(raw.image_url, `${label}.image_url`);
  return {
    imageName,
    imageUrl: publicPath(`/images/${encodeURIComponent(imageName)}`),
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

function parseOptionalProjectionResults(
  value: unknown,
  label: string,
  jointContext: JointHierarchyContext,
): OptionalProjectionResults {
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
    ...(raw.joint_hierarchy === undefined ? {} : {
      jointHierarchy: parseJointHierarchy(raw.joint_hierarchy, `${label}.joint_hierarchy`, jointContext),
    }),
  };
}

function parseBrainMapResults(value: unknown): BrainMapResults {
  const label = 'brain_map_results';
  const raw = object(value, label);
  if (raw.schema_version !== 1) {
    throw new CortexSessionError('invalid', `${label}.schema_version must be 1.`);
  }
  if (raw.subject_id !== 's1') {
    throw new CortexSessionError('invalid', `${label}.subject_id must be "s1".`);
  }
  if (raw.response_space !== 'nsd_1000_subject1_floc_t7_v1') {
    throw new CortexSessionError(
      'invalid',
      `${label}.response_space must be "nsd_1000_subject1_floc_t7_v1".`,
    );
  }

  const rawCounts = object(raw.roi_voxel_counts, `${label}.roi_voxel_counts`);
  const roiVoxelCounts = Object.fromEntries(CANONICAL_ROIS.map((roi) => {
    const count = integer(rawCounts[roi], `${label}.roi_voxel_counts.${roi}`);
    if (count !== SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]) {
      throw new CortexSessionError(
        'invalid',
        `${label}.roi_voxel_counts.${roi} must be ${SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]}.`,
      );
    }
    return [roi, count];
  })) as Record<CanonicalRoi, number>;

  const stimuli = array(raw.stimuli, `${label}.stimuli`).map((value, stimulusIndex) => {
    const stimulusLabel = `${label}.stimuli[${stimulusIndex}]`;
    const stimulus = object(value, stimulusLabel);
    const rawResponses = object(stimulus.roi_responses, `${stimulusLabel}.roi_responses`);
    const roiResponses = Object.fromEntries(CANONICAL_ROIS.map((roi) => {
      const expectedLength = roiVoxelCounts[roi];
      const responses = array(rawResponses[roi], `${stimulusLabel}.roi_responses.${roi}`).map(
        (response, responseIndex) => finite(
          response,
          `${stimulusLabel}.roi_responses.${roi}[${responseIndex}]`,
        ),
      );
      if (responses.length !== expectedLength) {
        throw new CortexSessionError(
          'invalid',
          `${stimulusLabel}.roi_responses.${roi} must contain ${expectedLength} responses.`,
        );
      }
      return [roi, responses];
    })) as Record<CanonicalRoi, number[]>;
    return {
      stimulusId: string(stimulus.stimulus_id, `${stimulusLabel}.stimulus_id`),
      roiResponses,
    };
  });

  return {
    schemaVersion: 1,
    subjectId: 's1',
    responseSpace: 'nsd_1000_subject1_floc_t7_v1',
    roiVoxelCounts,
    stimuli,
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
    throw new CortexSessionError('invalid', 'This Cortex-Compass session uses an unsupported schema version.');
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
    throw new CortexSessionError('expired', 'This Cortex-Compass session has expired. Return to Virtual Cortex to prepare a new one.');
  }

  const stimuli = array(raw.stimuli, 'stimuli').map((stimulus, index) => parseStimulus(stimulus, index, baseUrl));
  if (stimuli.length === 0 || new Set(stimuli.map((stimulus) => stimulus.id)).size !== stimuli.length) {
    throw new CortexSessionError('invalid', 'The session must contain at least one uniquely identified stimulus.');
  }

  const rawRoiResults = object(raw.roi_results, 'roi_results');
  if (Object.keys(rawRoiResults).length !== CANONICAL_ROIS.length
    || Object.keys(rawRoiResults).some((roi) => !CANONICAL_ROIS.includes(roi as CanonicalRoi))) {
    throw new CortexSessionError('invalid', 'ROI results must contain exactly FFA, PPA, and EBA.');
  }
  const jointContext: JointHierarchyContext = {
    referenceCount: comparison.reference_count,
    uploadedIds: stimuli.map((stimulus) => stimulus.id),
  };
  const roiResults = Object.fromEntries(
    CANONICAL_ROIS.map((roi) => [roi, parseRoiResult(rawRoiResults[roi], roi, baseUrl, jointContext)]),
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
    : parseOptionalProjectionResults(raw.visual_results, 'visual_results', jointContext);
  const dreamsimResults = raw.dreamsim_results === undefined || raw.dreamsim_results === null
    ? undefined
    : parseOptionalProjectionResults(raw.dreamsim_results, 'dreamsim_results', jointContext);
  for (const [label, results] of [['visual_results', visualResults], ['dreamsim_results', dreamsimResults]] as const) {
    if (!results) continue;
    const actualIds = results.stimuli.map((stimulus) => stimulus.stimulusId).sort();
    if (actualIds.length !== expectedIds.length || actualIds.some((id, index) => id !== expectedIds[index])) {
      throw new CortexSessionError('invalid', `${label} does not contain every uploaded stimulus exactly once.`);
    }
  }

  const jointHierarchies = [
    ...CANONICAL_ROIS.map((roi) => roiResults[roi].jointHierarchy),
    visualResults?.jointHierarchy,
    dreamsimResults?.jointHierarchy,
  ].filter((hierarchy): hierarchy is JointHierarchy => hierarchy !== undefined);
  const firstJointHierarchy = jointHierarchies[0];
  if (firstJointHierarchy) {
    const referenceIds = new Set(
      firstJointHierarchy.members.slice(0, firstJointHierarchy.referenceCount).map((member) => member.id),
    );
    for (const hierarchy of jointHierarchies.slice(1)) {
      if (hierarchy.members.slice(0, hierarchy.referenceCount).some((member) => !referenceIds.has(member.id))) {
        throw new CortexSessionError('invalid', 'Joint hierarchies do not contain the same reference image IDs.');
      }
    }
  }

  const brainMapResults = raw.brain_map_results === undefined || raw.brain_map_results === null
    ? undefined
    : parseBrainMapResults(raw.brain_map_results);
  if (brainMapResults) {
    const actualIds = brainMapResults.stimuli.map((stimulus) => stimulus.stimulusId).sort();
    if (actualIds.length !== expectedIds.length || actualIds.some((id, index) => id !== expectedIds[index])) {
      throw new CortexSessionError(
        'invalid',
        'brain_map_results does not contain every uploaded stimulus exactly once.',
      );
    }
  }

  const primaryModel = string(comparison.model, 'comparison.model');
  let modelResults: Record<string, SessionModelResults> | undefined;
  if (raw.model_results !== undefined && raw.model_results !== null) {
    const additionalResults = object(raw.model_results, 'model_results');
    const entries = Object.entries(additionalResults);
    if (entries.length > 3) {
      throw new CortexSessionError('invalid', 'A comparison session supports at most three additional models.');
    }
    let referenceOrder = firstJointHierarchy?.members.slice(0, 185).map((member) => member.id);
    modelResults = Object.fromEntries(entries.map(([model, value]) => {
      string(model, 'model_results model');
      if (model === primaryModel) {
        throw new CortexSessionError('invalid', 'Additional model results must not duplicate the primary model.');
      }
      const results = object(value, `model_results.${model}`);
      if (Object.keys(results).some((key) => key !== 'roi_results' && key !== 'brain_map_results')) {
        throw new CortexSessionError('invalid', `model_results.${model} contains unexpected fields.`);
      }
      // Validate with the exact same contracts as the primary model, sharing
      // image identity, expiration, reference version, and model-independent
      // visual features. Strip the model map to make this a bounded one-level
      // validation, never recursively accept nested model-result payloads.
      const scoped = validateComparisonSession({
        ...raw,
        comparison: { ...comparison, model },
        roi_results: results.roi_results,
        brain_map_results: results.brain_map_results,
        model_results: undefined,
      }, baseUrl, now);
      for (const roi of CANONICAL_ROIS) {
        const actualIds = scoped.roiResults[roi].stimuli.map((stimulus) => stimulus.stimulusId);
        if (actualIds.some((id, index) => id !== stimuli[index].id)) {
          throw new CortexSessionError('invalid', `model_results.${model}.${roi} must preserve uploaded stimulus order.`);
        }
        const hierarchy = scoped.roiResults[roi].jointHierarchy;
        if (!referenceOrder && hierarchy) {
          referenceOrder = hierarchy.members.slice(0, 185).map((member) => member.id);
        }
        if (referenceOrder && hierarchy
          && hierarchy.members.slice(0, 185).some((member, index) => member.id !== referenceOrder[index])) {
          throw new CortexSessionError('invalid', `model_results.${model}.${roi} must preserve reference image order.`);
        }
      }
      if (scoped.brainMapResults?.stimuli.some((stimulus, index) => stimulus.stimulusId !== stimuli[index].id)) {
        throw new CortexSessionError('invalid', `model_results.${model}.brain_map_results must preserve uploaded stimulus order.`);
      }
      return [model, {
        roiResults: scoped.roiResults,
        ...(scoped.brainMapResults ? { brainMapResults: scoped.brainMapResults } : {}),
      }];
    }));
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
      model: primaryModel,
      dataset: 'nsd_1000',
      rois: [...CANONICAL_ROIS],
      referenceVersion: string(comparison.reference_version, 'comparison.reference_version'),
      referenceCount: 185,
    },
    stimuli,
    roiResults,
    ...(visualResults ? { visualResults } : {}),
    ...(dreamsimResults ? { dreamsimResults } : {}),
    ...(brainMapResults ? { brainMapResults } : {}),
    ...(modelResults && Object.keys(modelResults).length > 0 ? { modelResults } : {}),
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
  if (manifest.referenceCount !== 185 || getSessionModels(session).some((model) => !manifest.models.includes(model))) {
    throw new CortexSessionError('version-mismatch', 'The local Cortex-Compass data does not support this comparison session.');
  }
}

async function readJsonResponse(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new CortexSessionError('invalid', 'The server returned malformed session data.');
  }
}

async function loadSingleComparisonSession(
  token: string,
  fetcher: typeof fetch,
): Promise<CortexComparisonSession> {
  const baseUrl = coreBaseUrl();
  let response: Response;
  try {
    response = await fetcher(`${baseUrl}/comparison-sessions/${encodeURIComponent(token)}`, {
      headers: { Accept: 'application/json' },
      method: 'GET',
    });
  } catch {
    throw new CortexSessionError('unavailable', 'Cortex Core could not be reached. Cortex-Compass remains available without this session.');
  }
  if (response.status === 410) {
    throw new CortexSessionError('expired', 'This Cortex-Compass session has expired. Return to Virtual Cortex to prepare a new one.');
  }
  if (response.status === 404) {
    throw new CortexSessionError('invalid', 'This Cortex-Compass session does not exist or is no longer available.');
  }
  if (!response.ok) {
    throw new CortexSessionError('unavailable', `Cortex Core could not load the session (HTTP ${response.status}).`);
  }
  return validateComparisonSession(await readJsonResponse(response), baseUrl);
}

function sessionHierarchies(session: CortexComparisonSession): JointHierarchy[] {
  return [
    ...CANONICAL_ROIS.map((roi) => session.roiResults[roi].jointHierarchy),
    session.visualResults?.jointHierarchy,
    session.dreamsimResults?.jointHierarchy,
  ].filter((hierarchy): hierarchy is JointHierarchy => hierarchy !== undefined);
}

function assertBundleCompatible(sessions: CortexComparisonSession[]): void {
  const primary = sessions[0];
  const models = new Set<string>();
  let referenceOrder: string[] | undefined;
  let referenceIds: Set<string> | undefined;
  for (const session of sessions) {
    if (session.modelResults) {
      throw new CortexSessionError('invalid', 'A multi-session handoff must contain single-model sessions only.');
    }
    if (models.has(session.comparison.model)) {
      throw new CortexSessionError('invalid', 'The comparison sessions contain duplicate models.');
    }
    models.add(session.comparison.model);
    if (session.sourceRequest.model !== primary.sourceRequest.model
      || session.sourceRequest.dataset !== primary.sourceRequest.dataset
      || session.sourceRequest.roi !== primary.sourceRequest.roi) {
      throw new CortexSessionError('invalid', 'The comparison sessions do not share the same original Lab settings.');
    }
    if (session.comparison.referenceVersion !== primary.comparison.referenceVersion
      || session.comparison.referenceCount !== primary.comparison.referenceCount
      || session.comparison.dataset !== primary.comparison.dataset
      || session.comparison.rois.some((roi, index) => roi !== primary.comparison.rois[index])) {
      throw new CortexSessionError('version-mismatch', 'The comparison sessions use different historical reference releases.');
    }
    if (session.stimuli.length !== primary.stimuli.length
      || session.stimuli.some((stimulus, index) => stimulus.displayName !== primary.stimuli[index].displayName)) {
      throw new CortexSessionError('invalid', 'The comparison sessions do not preserve the same image names and order.');
    }
    const ids = session.stimuli.map((stimulus) => stimulus.id);
    const orderedResults = [
      ...CANONICAL_ROIS.map((roi) => session.roiResults[roi].stimuli),
      session.brainMapResults?.stimuli,
      session.visualResults?.stimuli,
      session.dreamsimResults?.stimuli,
    ].filter((results) => results !== undefined);
    if (orderedResults.some((results) => results.some((stimulus, index) => stimulus.stimulusId !== ids[index]))) {
      throw new CortexSessionError('invalid', 'Each comparison session must preserve uploaded stimulus order in every result.');
    }
    for (const roi of CANONICAL_ROIS) {
      const hierarchy = session.roiResults[roi].jointHierarchy;
      if (!hierarchy) continue;
      const order = hierarchy.members.slice(0, hierarchy.referenceCount).map((member) => member.id);
      referenceOrder ??= order;
      if (order.some((id, index) => id !== referenceOrder![index])) {
        throw new CortexSessionError('invalid', 'The model joint hierarchies do not preserve reference image order.');
      }
    }
    for (const hierarchy of sessionHierarchies(session)) {
      const ids = hierarchy.members.slice(0, hierarchy.referenceCount).map((member) => member.id);
      referenceIds ??= new Set(ids);
      if (ids.some((id) => !referenceIds!.has(id))) {
        throw new CortexSessionError('invalid', 'The comparison sessions do not contain the same canonical reference image IDs.');
      }
    }
  }
}

async function verifyBundleImages(sessions: CortexComparisonSession[], fetcher: typeof fetch): Promise<void> {
  if (!globalThis.crypto?.subtle) {
    throw new CortexSessionError('configuration', 'A secure connection is required to verify comparison image identity.');
  }
  // Separate Core requests generate independent temporary IDs. Names alone do
  // not prove image identity (duplicate filenames are valid), so verify the
  // deterministic sanitized PNG bytes before aligning any model responses.
  const jobs = sessions.flatMap((session, sessionIndex) => session.stimuli.map((stimulus, imageIndex) => ({
    sessionIndex, imageIndex, imageUrl: stimulus.imageUrl,
  })));
  const digests = sessions.map((session) => new Array<string>(session.stimuli.length));
  let nextJob = 0;
  let verificationError: unknown;
  const abortController = new AbortController();
  const worker = async () => {
    while (!verificationError && nextJob < jobs.length) {
      const job = jobs[nextJob++];
      try {
        const response = await fetcher(job.imageUrl, {
          method: 'GET', headers: { Accept: 'image/png' }, signal: abortController.signal,
        });
        if (!response.ok) {
          throw new CortexSessionError(response.status === 410 ? 'expired' : 'unavailable',
            'A comparison image is no longer available. Return to Virtual Cortex to prepare a new comparison.');
        }
        const bytes = await response.arrayBuffer();
        const signature = new Uint8Array(bytes, 0, Math.min(bytes.byteLength, 8));
        if (signature.length !== 8 || [137, 80, 78, 71, 13, 10, 26, 10].some((byte, index) => signature[index] !== byte)) {
          throw new CortexSessionError('invalid', 'The server returned an invalid comparison image.');
        }
        const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
        digests[job.sessionIndex][job.imageIndex] = Array.from(new Uint8Array(digest),
          (byte) => byte.toString(16).padStart(2, '0')).join('');
      } catch (error) {
        verificationError ??= error instanceof CortexSessionError ? error
          : new CortexSessionError('unavailable', 'Cortex Core could not verify the comparison images.');
        abortController.abort();
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(4, jobs.length) }, worker));
  if (verificationError) throw verificationError;
  if (digests.slice(1).some((images) => images.some((digest, index) => digest !== digests[0][index]))) {
    throw new CortexSessionError('invalid', 'The comparison sessions contain different images or image order.');
  }
}

function remapJointHierarchy(hierarchy: JointHierarchy | undefined, ids: Map<string, string>): JointHierarchy | undefined {
  return hierarchy && {
    ...hierarchy,
    members: hierarchy.members.map((member) => member.kind === 'uploaded'
      ? { ...member, id: ids.get(member.id)! }
      : member),
  };
}

function combineSingleModelSessions(sessions: CortexComparisonSession[]): CortexComparisonSession {
  const primary = sessions[0];
  const expiresAt = sessions.reduce((earliest, session) => Date.parse(session.expiresAt) < Date.parse(earliest)
    ? session.expiresAt : earliest, primary.expiresAt);
  if (Date.parse(expiresAt) <= Date.now()) {
    throw new CortexSessionError('expired', 'This Cortex-Compass session has expired. Return to Virtual Cortex to prepare a new one.');
  }
  return {
    ...primary,
    expiresAt,
    modelResults: Object.fromEntries(sessions.slice(1).map((session) => {
      const ids = new Map(session.stimuli.map((stimulus, index) => [stimulus.id, primary.stimuli[index].id]));
      const roiResults = Object.fromEntries(CANONICAL_ROIS.map((roi) => [roi, {
        ...session.roiResults[roi],
        stimuli: session.roiResults[roi].stimuli.map((stimulus) => ({ ...stimulus, stimulusId: ids.get(stimulus.stimulusId)! })),
        ...(session.roiResults[roi].jointHierarchy ? {
          jointHierarchy: remapJointHierarchy(session.roiResults[roi].jointHierarchy, ids),
        } : {}),
      }])) as Record<CanonicalRoi, SessionRoiResult>;
      return [session.comparison.model, {
        roiResults,
        ...(session.brainMapResults ? { brainMapResults: {
          ...session.brainMapResults,
          stimuli: session.brainMapResults.stimuli.map((stimulus) => ({ ...stimulus, stimulusId: ids.get(stimulus.stimulusId)! })),
        } } : {}),
      }];
    })),
  };
}

export async function loadCortexComparisonSession(
  token: string,
  fetcher: typeof fetch = fetch,
): Promise<CortexComparisonSession> {
  const tokens = parseSessionTokens(token);
  // Single opaque tokens include legacy combined sessions and need no extra
  // image requests. The multi-session transport is only a frontend concern;
  // no combined endpoint or model_results backend field is required.
  const sessions = await Promise.all(tokens.map((part) => loadSingleComparisonSession(part, fetcher)));
  if (sessions.length === 1) return sessions[0];
  assertBundleCompatible(sessions);
  await verifyBundleImages(sessions, fetcher);
  return combineSingleModelSessions(sessions);
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
