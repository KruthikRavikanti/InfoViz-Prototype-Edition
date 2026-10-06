import { afterEach, describe, expect, it, vi } from 'vitest';
import manifestJson from '../../public/data/brain/reference/manifest.json';
import * as brainAssets from './brainMapAssets';
import {
  BRAIN_REFERENCE_MANIFEST_PATH,
  BRAIN_REFERENCE_VERSION,
  decodeBrainMapReferences,
  loadBrainMapReferences,
  loadBrainReferenceManifest,
  validateBrainReferenceManifest,
} from './brainReferenceAssets';

function responseBytes(): ArrayBuffer {
  const bytes = new ArrayBuffer(185 * 844 * 4);
  const values = new DataView(bytes);
  for (let row = 0; row < 185; row += 1) {
    for (let column = 0; column < 844; column += 1) values.setFloat32((row * 844 + column) * 4, row + column / 1024, true);
  }
  return bytes;
}

function fakeAssets() {
  const binary = responseBytes();
  const manifest = new TextEncoder().encode(JSON.stringify(manifestJson)).buffer;
  const fetcher = vi.fn<typeof fetch>();
  const verified = vi.spyOn(brainAssets, 'fetchVerifiedBrainAsset').mockImplementation(async (url) => (
    url === BRAIN_REFERENCE_MANIFEST_PATH ? manifest : binary
  ));
  return { binary, fetcher, verified };
}

afterEach(() => vi.restoreAllMocks());

describe('Murty-185 Subject 1 reference brain predictions', () => {
  it('accepts the pinned raw prediction identity and all audited model coverage', () => {
    const manifest = validateBrainReferenceManifest(manifestJson);
    expect(manifest.referenceVersion).toBe(BRAIN_REFERENCE_VERSION);
    expect(manifest.imageIds).toHaveLength(185);
    expect(Object.keys(manifest.models)).toHaveLength(26);
    expect(manifest.models.clip_rn50.url).toMatch(/\/data\/brain\/reference\/clip_rn50.f32$/);
  });

  it.each([
    ['dataset', 'murty_185'],
    ['referenceVersion', '0'.repeat(64)],
    ['subjectId', 's2'],
    ['normalization', 'zscore'],
    ['responseKind', 'measured_bold'],
    ['responseOrdering', 'xyz-c-order'],
    ['subjectColumnPolicy', 'width_only'],
    ['rowWidth', 843],
    ['referenceCount', 184],
    ['dtype', 'float32-be'],
  ])('rejects incompatible %s metadata', (field, value) => {
    const invalid = structuredClone(manifestJson) as Record<string, unknown>;
    invalid[field] = value;
    expect(() => validateBrainReferenceManifest(invalid)).toThrow(brainAssets.BrainMapAssetError);
  });

  it('rejects reordered images, altered voxel order, unsafe asset paths, and missing model coverage', () => {
    const badIds = structuredClone(manifestJson);
    [badIds.imageIds[0], badIds.imageIds[1]] = [badIds.imageIds[1], badIds.imageIds[0]];
    expect(() => validateBrainReferenceManifest(badIds)).toThrow(/identity\/order/);
    const badLayout = structuredClone(manifestJson);
    badLayout.roiLayout.eba.offset = 156;
    expect(() => validateBrainReferenceManifest(badLayout)).toThrow(/voxel identity\/order/);
    const badPath = structuredClone(manifestJson);
    badPath.models.clip_rn50.url = '../private-data.f32';
    expect(() => validateBrainReferenceManifest(badPath)).toThrow(/asset metadata/);
    const missing = structuredClone(manifestJson) as { models: Record<string, unknown> };
    delete missing.models.clip_rn50;
    expect(() => validateBrainReferenceManifest(missing)).toThrow(/coverage/);
  });

  it('decodes exact little-endian Subject 1 ROI boundaries with canonical image URLs', () => {
    const result = decodeBrainMapReferences(responseBytes(), validateBrainReferenceManifest(manifestJson), 'clip_rn50');
    expect(result.model).toBe('clip_rn50');
    expect(result.stimuli).toHaveLength(185);
    expect(result.stimuli[0].roiResponses.ffa).toHaveLength(156);
    expect(result.stimuli[0].roiResponses.ppa).toHaveLength(170);
    expect(result.stimuli[0].roiResponses.eba).toHaveLength(518);
    expect(result.stimuli[0].roiResponses.ffa[155]).toBe(155 / 1024);
    expect(result.stimuli[0].roiResponses.ppa[0]).toBe(156 / 1024);
    expect(result.stimuli[0].roiResponses.eba[0]).toBe(326 / 1024);
    expect(result.stimuli[184].roiResponses.eba[517]).toBe(184 + 843 / 1024);
    expect(result.stimuli[184].imageUrl).toMatch(/\/images\/image_185.png$/);
    expect(Object.isFrozen(result.stimuli[0].roiResponses.ffa)).toBe(true);
  });

  it('fails closed for truncated, non-finite, or unsupported response arrays', () => {
    const manifest = validateBrainReferenceManifest(manifestJson);
    expect(() => decodeBrainMapReferences(new ArrayBuffer(4), manifest, 'clip_rn50')).toThrow(/dimensions/);
    const nonFinite = responseBytes();
    new DataView(nonFinite).setFloat32(0, Number.NaN, true);
    expect(() => decodeBrainMapReferences(nonFinite, manifest, 'clip_rn50')).toThrow(/finite/);
    expect(() => decodeBrainMapReferences(responseBytes(), manifest, 'unknown')).toThrow(/unavailable/);
  });

  it('deduplicates concurrent requests, verifies the pinned manifest and model hashes, and reuses results', async () => {
    const { fetcher, verified } = fakeAssets();
    const [first, second] = await Promise.all([
      loadBrainMapReferences('clip_rn50', fetcher), loadBrainMapReferences('clip_rn50', fetcher),
    ]);
    expect(first).toBe(second);
    expect(await loadBrainMapReferences('clip_rn50', fetcher)).toBe(first);
    expect(verified).toHaveBeenCalledTimes(2);
    expect(verified.mock.calls[0][1]).toBe('86ac452a2c5badeb436a219be70e10eef47825b749bcf8abdf67b1b9d78b301d');
    expect(verified.mock.calls[1][1]).toBe(manifestJson.models.clip_rn50.sha256);
    expect(await loadBrainReferenceManifest(fetcher)).toHaveProperty('referenceVersion', BRAIN_REFERENCE_VERSION);
    expect(verified).toHaveBeenCalledTimes(2);
  });

  it('aborts a caller without cancelling another consumer or poisoning cached results', async () => {
    const { fetcher } = fakeAssets();
    const cancelled = new AbortController();
    const wait = loadBrainMapReferences('clip_rn50', fetcher, cancelled.signal);
    cancelled.abort();
    await expect(wait).rejects.toHaveProperty('name', 'AbortError');
    await expect(loadBrainMapReferences('clip_rn50', fetcher)).resolves.toHaveProperty('model', 'clip_rn50');
    const alreadyAborted = new AbortController();
    alreadyAborted.abort();
    await expect(loadBrainMapReferences('clip_rn50', fetcher, alreadyAborted.signal)).rejects.toHaveProperty('name', 'AbortError');
  });

  it('does not fetch unsupported models, and permits a retry after integrity failure', async () => {
    const { fetcher, verified } = fakeAssets();
    await expect(loadBrainMapReferences('unknown', fetcher)).rejects.toThrow(/unavailable/);
    expect(verified).not.toHaveBeenCalled();
    verified.mockRejectedValueOnce(new brainAssets.BrainMapAssetError('integrity check failed'));
    await expect(loadBrainMapReferences('clip_rn50', fetcher)).rejects.toThrow(/integrity/);
    await expect(loadBrainMapReferences('clip_rn50', fetcher)).resolves.toHaveProperty('model', 'clip_rn50');
  });
});
