import { REFERENCE_COVERAGE } from '../generated/referenceCoverage';
import type { BrainMapReferenceSet } from '../types/brainMap';
import { CANONICAL_ROIS, type CanonicalRoi } from '../types/session';
import { publicPath } from '../utils/publicPath';
import { BrainMapAssetError, fetchVerifiedBrainAsset } from './brainMapAssets';

export const BRAIN_REFERENCE_MANIFEST_PATH = publicPath('/data/brain/reference/manifest.json');
export const BRAIN_REFERENCE_VERSION = REFERENCE_COVERAGE.referenceVersion;
export const BRAIN_REFERENCE_MANIFEST_SHA256 = REFERENCE_COVERAGE.brainManifestSha256;
const RESPONSE_SPACE = 'nsd_1000_subject1_floc_t7_v1';
const REFERENCE_COUNT = 185;
const ROW_WIDTH = 844;
const BYTE_LENGTH = REFERENCE_COUNT * ROW_WIDTH * 4;
const COUNTS: Record<CanonicalRoi, number> = { ffa: 156, ppa: 170, eba: 518 };
const OFFSETS: Record<CanonicalRoi, number> = { ffa: 0, ppa: 156, eba: 326 };
const COORDINATE_HASHES: Record<CanonicalRoi, string> = {
  ffa: 'ce13c8bcd14cc54865b90cc4d9546559afd92269ad2bdc9a66df9aa327ef8a88',
  ppa: 'e0b164fe5a54d564b09e4dc25b4d456bb966933870e3dbb9aebd93d73657282e',
  eba: '440e3d36e725f44df7a4bfc6aeab9cc3fc180853adde80843439393698c36fbe',
};

export type BrainReferenceManifest = {
  referenceVersion: typeof BRAIN_REFERENCE_VERSION;
  responseSpace: typeof RESPONSE_SPACE;
  imageIds: string[];
  models: Record<string, { url: string; bytes: number; sha256: string }>;
};

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new BrainMapAssetError(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

export function validateBrainReferenceManifest(value: unknown): BrainReferenceManifest {
  const raw = record(value, 'Brain reference manifest');
  const exactMetadata = {
    schemaVersion: 1,
    dataset: 'nsd_1000',
    stimulusSet: 'Murty-185',
    responseKind: 'model_prediction',
    subjectId: 's1',
    responseSpace: RESPONSE_SPACE,
    referenceVersion: BRAIN_REFERENCE_VERSION,
    normalization: 'none',
    responseOrdering: 'zyx-c-order',
    subjectColumnPolicy: 'audited_s1_prefix_from_pooled_nsd_readout',
    dtype: 'float32-le',
    rowWidth: ROW_WIDTH,
    referenceCount: REFERENCE_COUNT,
  };
  for (const [field, expected] of Object.entries(exactMetadata)) {
    if (raw[field] !== expected) throw new BrainMapAssetError(`Brain reference manifest.${field} is incompatible.`);
  }
  const imageIds = Array.from({ length: REFERENCE_COUNT }, (_, index) => `image_${String(index + 1).padStart(3, '0')}.png`);
  if (!Array.isArray(raw.imageIds) || raw.imageIds.length !== REFERENCE_COUNT || raw.imageIds.some((id, index) => id !== imageIds[index])) {
    throw new BrainMapAssetError('Brain reference image identity/order must match all 185 canonical images.');
  }
  const layout = record(raw.roiLayout, 'Brain reference ROI layout');
  if (Object.keys(layout).length !== CANONICAL_ROIS.length) throw new BrainMapAssetError('Brain reference ROI layout must contain only FFA, PPA, and EBA.');
  for (const roi of CANONICAL_ROIS) {
    const entry = record(layout[roi], `Brain reference ${roi} layout`);
    if (entry.offset !== OFFSETS[roi] || entry.count !== COUNTS[roi] || entry.orderedCoordinateSha256 !== COORDINATE_HASHES[roi]) {
      throw new BrainMapAssetError(`Brain reference ${roi.toUpperCase()} voxel identity/order is incompatible.`);
    }
  }
  const rawModels = record(raw.models, 'Brain reference models');
  const expectedModels = REFERENCE_COVERAGE.models;
  // Exact reviewed roster still governs eligibility; allow the canonical
  // decimal TopoNets ID without admitting arbitrary dots or path traversal.
  const safeModelId = (model: string) => /^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(model) || model === 'toponets_rn18_tau_0.5';
  if (expectedModels.some((model) => !safeModelId(model)) || Object.keys(rawModels).length !== expectedModels.length || expectedModels.some((model) => !(model in rawModels))) {
    throw new BrainMapAssetError('Brain reference models must match the audited model coverage.');
  }
  const models: BrainReferenceManifest['models'] = {};
  for (const model of expectedModels) {
    const entry = record(rawModels[model], `Brain reference model ${model}`);
    if (entry.url !== `${model}.f32` || entry.bytes !== BYTE_LENGTH || typeof entry.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(entry.sha256)) {
      throw new BrainMapAssetError(`Brain reference model ${model} asset metadata is invalid.`);
    }
    models[model] = Object.freeze({ url: publicPath(`/data/brain/reference/${model}.f32`), bytes: BYTE_LENGTH, sha256: entry.sha256 });
  }
  Object.freeze(imageIds);
  Object.freeze(models);
  return Object.freeze({ referenceVersion: BRAIN_REFERENCE_VERSION, responseSpace: RESPONSE_SPACE, imageIds, models });
}

export function decodeBrainMapReferences(bytes: ArrayBuffer, manifest: BrainReferenceManifest, model: string): BrainMapReferenceSet {
  if (!manifest.models[model]) throw new BrainMapAssetError(`Reference brain predictions are unavailable for ${model}.`);
  if (manifest.referenceVersion !== BRAIN_REFERENCE_VERSION || manifest.responseSpace !== RESPONSE_SPACE || manifest.imageIds.length !== REFERENCE_COUNT) {
    throw new BrainMapAssetError('Brain reference identity is incompatible.');
  }
  if (bytes.byteLength !== BYTE_LENGTH) throw new BrainMapAssetError('Brain reference response dimensions are invalid.');
  const values = new DataView(bytes);
  const stimuli = manifest.imageIds.map((imageName, row) => {
    const roiResponses = Object.fromEntries(CANONICAL_ROIS.map((roi) => {
      const responses = Array.from({ length: COUNTS[roi] }, (_, column) => {
        const response = values.getFloat32((row * ROW_WIDTH + OFFSETS[roi] + column) * 4, true);
        if (!Number.isFinite(response)) throw new BrainMapAssetError('Brain reference predictions must be finite.');
        return response;
      });
      Object.freeze(responses);
      return [roi, responses];
    })) as Record<CanonicalRoi, number[]>;
    Object.freeze(roiResponses);
    return Object.freeze({ imageName, imageUrl: publicPath(`/images/${imageName}`), roiResponses });
  });
  Object.freeze(stimuli);
  return Object.freeze({ model, referenceVersion: manifest.referenceVersion, responseSpace: manifest.responseSpace, stimuli });
}

type Cache = { manifest?: Promise<BrainReferenceManifest>; models: Map<string, Promise<BrainMapReferenceSet>> };
const caches = new WeakMap<typeof fetch, Cache>();

function cacheFor(fetcher: typeof fetch): Cache {
  let cache = caches.get(fetcher);
  if (!cache) {
    cache = { models: new Map() };
    caches.set(fetcher, cache);
  }
  return cache;
}

function abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(new DOMException('The request was aborted.', 'AbortError'));
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(new DOMException('The request was aborted.', 'AbortError'));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
  });
}

/** Callers cancel their own wait; shared validated asset requests remain reusable. */
export function loadBrainReferenceManifest(fetcher: typeof fetch = fetch, signal?: AbortSignal): Promise<BrainReferenceManifest> {
  if (signal?.aborted) return Promise.reject(new DOMException('The request was aborted.', 'AbortError'));
  const cache = cacheFor(fetcher);
  if (!cache.manifest) {
    cache.manifest = fetchVerifiedBrainAsset(BRAIN_REFERENCE_MANIFEST_PATH, BRAIN_REFERENCE_MANIFEST_SHA256, 'Brain reference manifest', fetcher)
      .then((bytes) => {
        let value: unknown;
        try { value = JSON.parse(new TextDecoder().decode(bytes)); }
        catch { throw new BrainMapAssetError('Brain reference manifest contains malformed JSON.'); }
        return validateBrainReferenceManifest(value);
      })
      .catch((error: unknown) => { cache.manifest = undefined; throw error; });
  }
  return abortable(cache.manifest, signal);
}

export function loadBrainMapReferences(model: string, fetcher: typeof fetch = fetch, signal?: AbortSignal): Promise<BrainMapReferenceSet> {
  if (signal?.aborted) return Promise.reject(new DOMException('The request was aborted.', 'AbortError'));
  if (!REFERENCE_COVERAGE.models.includes(model)) {
    return Promise.reject(new BrainMapAssetError(`Reference brain predictions are unavailable for ${model}.`));
  }
  const cache = cacheFor(fetcher);
  let promise = cache.models.get(model);
  if (!promise) {
    promise = loadBrainReferenceManifest(fetcher).then(async (manifest) => {
      const entry = manifest.models[model];
      const bytes = await fetchVerifiedBrainAsset(entry.url, entry.sha256, `${model} brain reference predictions`, fetcher);
      return decodeBrainMapReferences(bytes, manifest, model);
    }).catch((error: unknown) => { cache.models.delete(model); throw error; });
    cache.models.set(model, promise);
  }
  return abortable(promise, signal);
}
