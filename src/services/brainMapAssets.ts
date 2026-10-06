import { SUBJECT_ONE_ROI_VOXEL_COUNTS, CANONICAL_ROIS, type CanonicalRoi } from '../types/session';
import type {
  BrainHemisphere,
  BrainMapAssets,
  BrainMapManifest,
  BrainProjection,
  BrainProjectionMapping,
  BrainProjectionRoi,
  BrainSurfaceAsset,
  BrainSurfaceKind,
} from '../types/brainMap';
import { publicPath } from '../utils/publicPath';

export const BRAIN_MAP_MANIFEST_PATH = publicPath('/data/brain/nsd-subj01/manifest.json');
const RESPONSE_SPACE = 'nsd_1000_subject1_floc_t7_v1';
const VOLUME_SHAPE_XYZ = [83, 104, 81] as const;
const EXPECTED_MAPPED_RESPONSE_COUNTS: Record<CanonicalRoi, number> = {
  ffa: 156,
  ppa: 170,
  eba: 515,
};
const EXPECTED_MANIFEST_SHA256 = '3ac10989e3ac26e8388d83b47d200a80d99a0b4532ff90386788b3285700f7ce';
const EXPECTED_PROJECTION_SHA256 = '4c0bf0d19aa6ceda5ae61f352f26b874f32450f49d0945b7c1022aaaa703618f';
const EXPECTED_SURFACE_SHA256: Record<BrainSurfaceKind, string> = {
  inflated: '62d9077896094e5ea196058379c2f2e85d6f5749207aa677c7355b5959a18241',
  pial: '8713d5d2f8f8ec24b42a65d9bcf9616561a942624610357d85224495b9c4f577',
};
const EXPECTED_COORDINATE_SHA256: Record<CanonicalRoi, string> = {
  ffa: 'ce13c8bcd14cc54865b90cc4d9546559afd92269ad2bdc9a66df9aa327ef8a88',
  ppa: 'e0b164fe5a54d564b09e4dc25b4d456bb966933870e3dbb9aebd93d73657282e',
  eba: '440e3d36e725f44df7a4bfc6aeab9cc3fc180853adde80843439393698c36fbe',
};

type JsonObject = Record<string, unknown>;

export class BrainMapAssetError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BrainMapAssetError';
  }
}

function object(value: unknown, label: string): JsonObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new BrainMapAssetError(`${label} must be an object.`);
  }
  return value as JsonObject;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new BrainMapAssetError(`${label} must be an array.`);
  return value;
}

function string(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new BrainMapAssetError(`${label} must be a non-empty string.`);
  }
  return value;
}

function integer(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new BrainMapAssetError(`${label} must be an integer.`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new BrainMapAssetError(`${label} must be a finite number.`);
  }
  return value;
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new BrainMapAssetError(`${label} must be a boolean.`);
  return value;
}

function color(value: unknown, label: string): string {
  const parsed = string(value, label);
  if (!/^#[\da-f]{6}$/i.test(parsed)) {
    throw new BrainMapAssetError(`${label} must be a six-digit hexadecimal color.`);
  }
  return parsed;
}

function sha256(value: unknown, label: string): string {
  const parsed = string(value, label).toLowerCase();
  if (!/^[\da-f]{64}$/.test(parsed)) throw new BrainMapAssetError(`${label} must be a SHA-256 digest.`);
  return parsed;
}

function resolveAssetUrl(path: string, manifestUrl: string): string {
  const browserOrigin = typeof window === 'undefined' ? 'http://localhost/' : window.location.href;
  const baseUrl = new URL(manifestUrl, browserOrigin);
  return new URL(path, baseUrl).toString();
}

function validateCompatibilityMetadata(raw: JsonObject, label: string) {
  if (raw.schemaVersion !== 1) throw new BrainMapAssetError(`${label}.schemaVersion must be 1.`);
  if (raw.subjectId !== 's1') throw new BrainMapAssetError(`${label}.subjectId must be "s1".`);
  if (raw.responseSpace !== RESPONSE_SPACE) {
    throw new BrainMapAssetError(`${label}.responseSpace is incompatible with this session.`);
  }
  if (raw.fLocThreshold !== 7 || raw.fLocThresholdOperator !== '>') {
    throw new BrainMapAssetError(`${label} must describe the Subject 1 fLoc selection at t > 7.`);
  }
  if (raw.responseOrdering !== 'zyx-c-order') {
    throw new BrainMapAssetError(`${label}.responseOrdering must be "zyx-c-order".`);
  }
  if (raw.mapper !== 'linenn') {
    throw new BrainMapAssetError(`${label}.mapper must be "linenn".`);
  }
  if (raw.normalizationPolicy !== 'per-surface-vertex') {
    throw new BrainMapAssetError(`${label}.normalizationPolicy must be "per-surface-vertex".`);
  }
  const rawShape = array(raw.volumeShapeXyz, `${label}.volumeShapeXyz`);
  if (
    rawShape.length !== VOLUME_SHAPE_XYZ.length
    || rawShape.some((dimension, index) => dimension !== VOLUME_SHAPE_XYZ[index])
  ) {
    throw new BrainMapAssetError(`${label}.volumeShapeXyz must be [83, 104, 81].`);
  }
}

function parseSurface(value: unknown, label: string, manifestUrl: string): BrainSurfaceAsset {
  const raw = object(value, label);
  const rawHemispheres = object(raw.hemispheres, `${label}.hemispheres`);
  const hemispheres = Object.fromEntries((['lh', 'rh'] as const).map((hemisphere) => {
    const rawHemisphere = object(rawHemispheres[hemisphere], `${label}.hemispheres.${hemisphere}`);
    const vertexCount = integer(rawHemisphere.vertexCount, `${label}.hemispheres.${hemisphere}.vertexCount`);
    const faceCount = integer(rawHemisphere.faceCount, `${label}.hemispheres.${hemisphere}.faceCount`);
    if (vertexCount <= 0 || faceCount <= 0) {
      throw new BrainMapAssetError(`${label}.hemispheres.${hemisphere} has an invalid mesh size.`);
    }
    return [hemisphere, {
      node: string(rawHemisphere.node, `${label}.hemispheres.${hemisphere}.node`),
      vertexCount,
      faceCount,
    }];
  })) as BrainSurfaceAsset['hemispheres'];
  return {
    url: resolveAssetUrl(string(raw.url, `${label}.url`), manifestUrl),
    sha256: sha256(raw.sha256, `${label}.sha256`),
    normalsIncluded: boolean(raw.normalsIncluded, `${label}.normalsIncluded`),
    hemispheres,
  };
}

export function validateBrainMapManifest(value: unknown, manifestUrl: string): BrainMapManifest {
  const raw = object(value, 'brain-map manifest');
  validateCompatibilityMetadata(raw, 'brain-map manifest');
  const subject = object(raw.subject, 'brain-map manifest.subject');
  if (subject.id !== 'nsd-subj01') {
    throw new BrainMapAssetError('brain-map manifest.subject.id must be "nsd-subj01".');
  }
  if (subject.responseSpace !== 'nsd_1000_subject1_floc_t7_v1') {
    throw new BrainMapAssetError('The brain-map manifest response space does not match this session.');
  }
  const volumeSpace = object(raw.volumeSpace, 'brain-map manifest.volumeSpace');
  const rawShape = array(volumeSpace.shapeXyz, 'brain-map manifest.volumeSpace.shapeXyz');
  if (rawShape.length !== 3) {
    throw new BrainMapAssetError('brain-map manifest.volumeSpace.shapeXyz must contain three dimensions.');
  }
  const shapeXyz = rawShape.map((dimension, index) => {
    const parsed = integer(dimension, `brain-map manifest.volumeSpace.shapeXyz[${index}]`);
    if (parsed <= 0) throw new BrainMapAssetError('Brain-map volume dimensions must be positive.');
    return parsed;
  }) as [number, number, number];
  const sourceAxisOrder = string(volumeSpace.sourceAxisOrder, 'brain-map manifest.volumeSpace.sourceAxisOrder');
  if (
    rawShape.some((dimension, index) => dimension !== VOLUME_SHAPE_XYZ[index])
    || sourceAxisOrder !== 'zyx'
  ) {
    throw new BrainMapAssetError('The nested NSD volume-space metadata is incompatible.');
  }

  const rawSurfaces = object(raw.surfaces, 'brain-map manifest.surfaces');
  const surfaces: Partial<Record<BrainSurfaceKind, BrainSurfaceAsset>> = {};
  for (const kind of ['inflated', 'pial'] as const) {
    if (rawSurfaces[kind] !== undefined && rawSurfaces[kind] !== null) {
      surfaces[kind] = parseSurface(rawSurfaces[kind], `brain-map manifest.surfaces.${kind}`, manifestUrl);
    }
  }
  if (!surfaces.inflated || !surfaces.pial) {
    throw new BrainMapAssetError('The brain-map manifest must contain both inflated and pial surfaces.');
  }
  for (const kind of ['inflated', 'pial'] as const) {
    if (surfaces[kind]!.sha256 !== EXPECTED_SURFACE_SHA256[kind]) {
      throw new BrainMapAssetError(`The ${kind} surface digest does not match the validated artifact.`);
    }
  }
  const expectedTopology = {
    lh: { node: 'lh', vertexCount: 227021, faceCount: 454038 },
    rh: { node: 'rh', vertexCount: 226601, faceCount: 453198 },
  } as const;
  for (const hemisphere of ['lh', 'rh'] as const) {
    const inflated = surfaces.inflated.hemispheres[hemisphere];
    const pial = surfaces.pial.hemispheres[hemisphere];
    if (inflated.vertexCount !== pial.vertexCount || inflated.faceCount !== pial.faceCount) {
      throw new BrainMapAssetError(
        `${hemisphere.toUpperCase()} pial and inflated surfaces must have identical topology.`,
      );
    }
    for (const [kind, entry] of [['inflated', inflated], ['pial', pial]] as const) {
      const expected = expectedTopology[hemisphere];
      if (
        entry.node !== expected.node
        || entry.vertexCount !== expected.vertexCount
        || entry.faceCount !== expected.faceCount
      ) {
        throw new BrainMapAssetError(
          `${hemisphere.toUpperCase()} ${kind} topology does not match the validated Subject 1 surface.`,
        );
      }
    }
  }

  const rawRois = object(raw.rois, 'brain-map manifest.rois');
  const rois = Object.fromEntries(CANONICAL_ROIS.map((roi) => {
    const entry = object(rawRois[roi], `brain-map manifest.rois.${roi}`);
    const responseCount = integer(entry.responseCount, `brain-map manifest.rois.${roi}.responseCount`);
    if (responseCount !== SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]) {
      throw new BrainMapAssetError(
        `brain-map manifest.rois.${roi}.responseCount must be ${SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]}.`,
      );
    }
    const mappedResponseCount = integer(
      entry.mappedResponseCount,
      `brain-map manifest.rois.${roi}.mappedResponseCount`,
    );
    if (mappedResponseCount < 0 || mappedResponseCount > responseCount) {
      throw new BrainMapAssetError(`brain-map manifest.rois.${roi}.mappedResponseCount is out of range.`);
    }
    if (mappedResponseCount !== EXPECTED_MAPPED_RESPONSE_COUNTS[roi]) {
      throw new BrainMapAssetError(
        `brain-map manifest.rois.${roi}.mappedResponseCount must be ${EXPECTED_MAPPED_RESPONSE_COUNTS[roi]}.`,
      );
    }
    const unmappedResponseIndices = parseIntegerArray(
      entry.unmappedResponseIndices,
      `brain-map manifest.rois.${roi}.unmappedResponseIndices`,
    );
    if (
      unmappedResponseIndices.length !== responseCount - mappedResponseCount
      || new Set(unmappedResponseIndices).size !== unmappedResponseIndices.length
      || unmappedResponseIndices.some((index) => index < 0 || index >= responseCount)
    ) {
      throw new BrainMapAssetError(`brain-map manifest.rois.${roi} has invalid unmapped response indices.`);
    }
    if (entry.threshold !== 7 || entry.thresholdOperator !== '>') {
      throw new BrainMapAssetError(`brain-map manifest.rois.${roi} must use the strict t > 7 selection.`);
    }
    const orderedCoordinateSha256 = sha256(
      entry.orderedCoordinateSha256,
      `brain-map manifest.rois.${roi}.orderedCoordinateSha256`,
    );
    if (orderedCoordinateSha256 !== EXPECTED_COORDINATE_SHA256[roi]) {
      throw new BrainMapAssetError(`brain-map manifest.rois.${roi} has an unrecognized coordinate order.`);
    }
    return [roi, {
      label: string(entry.label, `brain-map manifest.rois.${roi}.label`),
      responseCount,
      mappedResponseCount,
      unmappedResponseIndices,
      orderedCoordinateSha256,
      color: color(entry.color, `brain-map manifest.rois.${roi}.color`),
    }];
  })) as BrainMapManifest['rois'];
  const rawMeshSpace = object(raw.meshSpace, 'brain-map manifest.meshSpace');
  const meshSpace = {
    source: string(rawMeshSpace.source, 'brain-map manifest.meshSpace.source'),
    gltfTransform: string(rawMeshSpace.gltfTransform, 'brain-map manifest.meshSpace.gltfTransform'),
    mapper: string(rawMeshSpace.mapper, 'brain-map manifest.meshSpace.mapper'),
    vertexOrderingPreserved: boolean(
      rawMeshSpace.vertexOrderingPreserved,
      'brain-map manifest.meshSpace.vertexOrderingPreserved',
    ),
    recentered: boolean(rawMeshSpace.recentered, 'brain-map manifest.meshSpace.recentered'),
  };
  if (
    meshSpace.mapper !== 'pycortex linenn'
    || !meshSpace.vertexOrderingPreserved
    || meshSpace.recentered
  ) {
    throw new BrainMapAssetError('The Subject 1 mesh-space mapping metadata is incompatible.');
  }

  return {
    schemaVersion: 1,
    subjectId: 's1',
    responseSpace: RESPONSE_SPACE,
    fLocThreshold: 7,
    fLocThresholdOperator: '>',
    responseOrdering: 'zyx-c-order',
    mapper: 'linenn',
    normalizationPolicy: 'per-surface-vertex',
    volumeShapeXyz: [...VOLUME_SHAPE_XYZ],
    subject: {
      id: 'nsd-subj01',
      label: string(subject.label, 'brain-map manifest.subject.label'),
      responseSpace: 'nsd_1000_subject1_floc_t7_v1',
    },
    volumeSpace: {
      name: string(volumeSpace.name, 'brain-map manifest.volumeSpace.name'),
      shapeXyz,
      sourceAxisOrder,
      responseOrdering: string(
        volumeSpace.responseOrdering,
        'brain-map manifest.volumeSpace.responseOrdering',
      ),
    },
    meshSpace,
    surfaces,
    projectionUrl: resolveAssetUrl(
      string(raw.projectionUrl, 'brain-map manifest.projectionUrl'),
      manifestUrl,
    ),
    projectionSha256: (() => {
      const digest = sha256(raw.projectionSha256, 'brain-map manifest.projectionSha256');
      if (digest !== EXPECTED_PROJECTION_SHA256) {
        throw new BrainMapAssetError('The surface projection digest does not match the validated artifact.');
      }
      return digest;
    })(),
    rois,
  };
}

function parseIntegerArray(value: unknown, label: string): number[] {
  return array(value, label).map((entry, index) => integer(entry, `${label}[${index}]`));
}

function parseMapping(
  value: unknown,
  label: string,
  responseCount: number,
  vertexCount: number,
): BrainProjectionMapping {
  const raw = object(value, label);
  const responseIndices = parseIntegerArray(raw.responseIndices, `${label}.responseIndices`);
  const vertexIndices = parseIntegerArray(raw.vertexIndices, `${label}.vertexIndices`);
  const weights = array(raw.weights, `${label}.weights`).map((weight, index) => finite(weight, `${label}.weights[${index}]`));
  if (responseIndices.length !== vertexIndices.length || responseIndices.length !== weights.length) {
    throw new BrainMapAssetError(`${label} arrays must have equal lengths.`);
  }
  responseIndices.forEach((responseIndex, index) => {
    if (responseIndex < 0 || responseIndex >= responseCount) {
      throw new BrainMapAssetError(`${label}.responseIndices[${index}] is out of range.`);
    }
    if (vertexIndices[index] < 0 || vertexIndices[index] >= vertexCount) {
      throw new BrainMapAssetError(`${label}.vertexIndices[${index}] is out of range.`);
    }
    if (weights[index] < 0) {
      throw new BrainMapAssetError(`${label}.weights[${index}] must be non-negative.`);
    }
  });
  const weightSums = new Map<number, number>();
  vertexIndices.forEach((vertexIndex, index) => {
    weightSums.set(vertexIndex, (weightSums.get(vertexIndex) ?? 0) + weights[index]);
  });
  for (const [vertexIndex, sum] of weightSums) {
    if (Math.abs(sum - 1) > 1e-5) {
      throw new BrainMapAssetError(
        `${label} weights reaching vertex ${vertexIndex} must sum to 1 (received ${sum}).`,
      );
    }
  }
  return { responseIndices, vertexIndices, weights };
}

function parseProjectionRoi(
  value: unknown,
  roi: CanonicalRoi,
  manifest: BrainMapManifest,
): BrainProjectionRoi {
  const label = `brain-map projection.rois.${roi}`;
  const raw = object(value, label);
  const responseCount = manifest.rois[roi].responseCount;
  const declaredResponseCount = integer(raw.responseCount, `${label}.responseCount`);
  const declaredMappedResponseCount = integer(raw.mappedResponseCount, `${label}.mappedResponseCount`);
  if (declaredResponseCount !== responseCount) {
    throw new BrainMapAssetError(`${label}.responseCount does not match the manifest.`);
  }
  if (declaredMappedResponseCount !== manifest.rois[roi].mappedResponseCount) {
    throw new BrainMapAssetError(`${label}.mappedResponseCount does not match the manifest.`);
  }
  const volumeCoordinatesZyx = array(raw.volumeCoordinatesZyx, `${label}.volumeCoordinatesZyx`).map(
    (coordinate, responseIndex) => {
      const entries = array(coordinate, `${label}.volumeCoordinatesZyx[${responseIndex}]`);
      if (entries.length !== 3) {
        throw new BrainMapAssetError(`${label}.volumeCoordinatesZyx[${responseIndex}] must contain z, y, x.`);
      }
      return entries.map((entry, axis) => integer(
        entry,
        `${label}.volumeCoordinatesZyx[${responseIndex}][${axis}]`,
      )) as [number, number, number];
    },
  );
  const volumeLinearIndicesXyz = parseIntegerArray(
    raw.volumeLinearIndicesXyz,
    `${label}.volumeLinearIndicesXyz`,
  );
  const hemispheres = array(raw.hemispheres, `${label}.hemispheres`).map((value, index) => {
    if (value !== 'lh' && value !== 'rh' && value !== 'bilateral' && value !== 'unmapped') {
      throw new BrainMapAssetError(`${label}.hemispheres[${index}] is invalid.`);
    }
    return value;
  });
  if (
    volumeCoordinatesZyx.length !== responseCount
    || volumeLinearIndicesXyz.length !== responseCount
    || hemispheres.length !== responseCount
  ) {
    throw new BrainMapAssetError(`${label} metadata must contain ${responseCount} response entries.`);
  }
  volumeCoordinatesZyx.forEach(([z, y, x], responseIndex) => {
    if (z < 0 || z >= VOLUME_SHAPE_XYZ[2] || y < 0 || y >= VOLUME_SHAPE_XYZ[1] || x < 0 || x >= VOLUME_SHAPE_XYZ[0]) {
      throw new BrainMapAssetError(`${label}.volumeCoordinatesZyx[${responseIndex}] is outside the NSD volume.`);
    }
    const linearIndex = volumeLinearIndicesXyz[responseIndex];
    if (linearIndex < 0 || linearIndex >= VOLUME_SHAPE_XYZ[0] * VOLUME_SHAPE_XYZ[1] * VOLUME_SHAPE_XYZ[2]) {
      throw new BrainMapAssetError(`${label}.volumeLinearIndicesXyz[${responseIndex}] is out of range.`);
    }
    const expectedLinearIndex = x * VOLUME_SHAPE_XYZ[1] * VOLUME_SHAPE_XYZ[2] + y * VOLUME_SHAPE_XYZ[2] + z;
    if (linearIndex !== expectedLinearIndex) {
      throw new BrainMapAssetError(
        `${label}.volumeLinearIndicesXyz[${responseIndex}] does not match its z, y, x coordinate.`,
      );
    }
  });
  if (new Set(volumeLinearIndicesXyz).size !== volumeLinearIndicesXyz.length) {
    throw new BrainMapAssetError(`${label}.volumeLinearIndicesXyz contains duplicate voxels.`);
  }

  const rawMappings = object(raw.mappings, `${label}.mappings`);
  const referenceSurface = manifest.surfaces.inflated ?? manifest.surfaces.pial!;
  const mappings = Object.fromEntries((['lh', 'rh'] as const).map((hemisphere) => [
    hemisphere,
    parseMapping(
      rawMappings[hemisphere],
      `${label}.mappings.${hemisphere}`,
      responseCount,
      referenceSurface.hemispheres[hemisphere].vertexCount,
    ),
  ])) as Record<BrainHemisphere, BrainProjectionMapping>;
  const unmappedResponseIndices = parseIntegerArray(
    raw.unmappedResponseIndices,
    `${label}.unmappedResponseIndices`,
  );
  if (unmappedResponseIndices.some((index) => index < 0 || index >= responseCount)) {
    throw new BrainMapAssetError(`${label}.unmappedResponseIndices contains an out-of-range index.`);
  }
  if (new Set(unmappedResponseIndices).size !== unmappedResponseIndices.length) {
    throw new BrainMapAssetError(`${label}.unmappedResponseIndices contains duplicates.`);
  }
  if (
    unmappedResponseIndices.length !== manifest.rois[roi].unmappedResponseIndices.length
    || unmappedResponseIndices.some((index, position) => index !== manifest.rois[roi].unmappedResponseIndices[position])
  ) {
    throw new BrainMapAssetError(`${label}.unmappedResponseIndices does not match the manifest.`);
  }
  const mappedResponseIndices = new Set([
    ...mappings.lh.responseIndices,
    ...mappings.rh.responseIndices,
  ]);
  const unmappedResponseSet = new Set(unmappedResponseIndices);
  if (mappedResponseIndices.size !== manifest.rois[roi].mappedResponseCount) {
    throw new BrainMapAssetError(`${label} has an inconsistent mapped response count.`);
  }
  for (let responseIndex = 0; responseIndex < responseCount; responseIndex += 1) {
    const mapsLeft = mappings.lh.responseIndices.includes(responseIndex);
    const mapsRight = mappings.rh.responseIndices.includes(responseIndex);
    const mapped = mapsLeft || mapsRight;
    const unmapped = unmappedResponseSet.has(responseIndex);
    if (mapped === unmapped) {
      throw new BrainMapAssetError(`${label} response ${responseIndex} must be mapped or explicitly unmapped.`);
    }
    const expectedHemisphere = mapsLeft && mapsRight
      ? 'bilateral'
      : mapsLeft
        ? 'lh'
        : mapsRight
          ? 'rh'
          : 'unmapped';
    if (hemispheres[responseIndex] !== expectedHemisphere) {
      throw new BrainMapAssetError(
        `${label}.hemispheres[${responseIndex}] must be "${expectedHemisphere}" for its mapping coverage.`,
      );
    }
  }

  return {
    responseCount,
    mappedResponseCount: declaredMappedResponseCount,
    volumeCoordinatesZyx,
    volumeLinearIndicesXyz,
    unmappedResponseIndices,
    hemispheres,
    mappings,
  };
}

export function validateBrainProjection(value: unknown, manifest: BrainMapManifest): BrainProjection {
  const raw = object(value, 'brain-map projection');
  validateCompatibilityMetadata(raw, 'brain-map projection');
  const rawRois = object(raw.rois, 'brain-map projection.rois');
  return {
    schemaVersion: 1,
    subjectId: 's1',
    responseSpace: RESPONSE_SPACE,
    fLocThreshold: 7,
    fLocThresholdOperator: '>',
    responseOrdering: 'zyx-c-order',
    mapper: 'linenn',
    normalizationPolicy: 'per-surface-vertex',
    volumeShapeXyz: [...VOLUME_SHAPE_XYZ],
    rois: Object.fromEntries(CANONICAL_ROIS.map((roi) => [
      roi,
      parseProjectionRoi(rawRois[roi], roi, manifest),
    ])) as BrainProjection['rois'],
  };
}

function parseJsonBytes(bytes: ArrayBuffer, label: string): unknown {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new BrainMapAssetError(`${label} contains malformed JSON.`);
  }
}

async function digestHex(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new BrainMapAssetError('This browser cannot verify brain-map asset integrity.');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function fetchVerifiedBrainAsset(
  url: string,
  expectedSha256: string,
  label: string,
  fetcher: typeof fetch = fetch,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetcher(url, { signal });
  } catch {
    throw new BrainMapAssetError(`${label} could not be reached.`);
  }
  if (!response.ok) throw new BrainMapAssetError(`${label} could not be loaded (HTTP ${response.status}).`);
  const bytes = await response.arrayBuffer();
  const actualSha256 = await digestHex(bytes);
  if (actualSha256 !== expectedSha256.toLowerCase()) {
    throw new BrainMapAssetError(`${label} failed its SHA-256 integrity check.`);
  }
  return bytes;
}

export async function loadBrainMapAssets(
  fetcher: typeof fetch = fetch,
  signal?: AbortSignal,
): Promise<BrainMapAssets> {
  const browserOrigin = typeof window === 'undefined' ? 'http://localhost/' : window.location.href;
  const manifestUrl = new URL(BRAIN_MAP_MANIFEST_PATH, browserOrigin).toString();
  let manifestResponse: Response;
  try {
    manifestResponse = await fetcher(manifestUrl, { headers: { Accept: 'application/json' }, signal });
  } catch {
    throw new BrainMapAssetError('Subject 1 surface metadata could not be reached.');
  }
  if (!manifestResponse.ok) {
    throw new BrainMapAssetError(`Subject 1 surface metadata could not be loaded (HTTP ${manifestResponse.status}).`);
  }
  const manifestBytes = await manifestResponse.arrayBuffer();
  if (await digestHex(manifestBytes) !== EXPECTED_MANIFEST_SHA256) {
    throw new BrainMapAssetError('Subject 1 surface metadata failed its SHA-256 integrity check.');
  }
  const manifest = validateBrainMapManifest(parseJsonBytes(manifestBytes, 'Subject 1 surface metadata'), manifestUrl);

  let projectionResponse: Response;
  try {
    projectionResponse = await fetcher(manifest.projectionUrl, { headers: { Accept: 'application/json' }, signal });
  } catch {
    throw new BrainMapAssetError('Subject 1 surface projection could not be reached.');
  }
  if (!projectionResponse.ok) {
    throw new BrainMapAssetError(`Subject 1 surface projection could not be loaded (HTTP ${projectionResponse.status}).`);
  }
  const projectionBytes = await projectionResponse.arrayBuffer();
  if (await digestHex(projectionBytes) !== manifest.projectionSha256) {
    throw new BrainMapAssetError('Subject 1 surface projection failed its SHA-256 integrity check.');
  }
  const projection = validateBrainProjection(
    parseJsonBytes(projectionBytes, 'Subject 1 surface projection'),
    manifest,
  );
  return { manifest, projection };
}
