import { describe, expect, it } from 'vitest';
import manifestJson from '../../public/data/brain/nsd-subj01/manifest.json';
import projectionJson from '../../public/data/brain/nsd-subj01/projection.json';
import {
  BrainMapAssetError,
  fetchVerifiedBrainAsset,
  validateBrainMapManifest,
  validateBrainProjection,
} from './brainMapAssets';

describe('Subject 1 brain-map artifacts', () => {
  it('validates the checked-in anatomy and response-index projection as one contract', () => {
    const manifest = validateBrainMapManifest(
      manifestJson,
      'https://infoviz.example/data/brain/nsd-subj01/manifest.json',
    );
    const projection = validateBrainProjection(projectionJson, manifest);

    expect(manifest.surfaces.inflated?.hemispheres.lh.vertexCount).toBe(227021);
    expect(manifest.surfaces.pial?.hemispheres.rh.vertexCount).toBe(226601);
    expect(projection.rois.eba.mappedResponseCount).toBe(515);
    expect(projection.rois.eba.unmappedResponseIndices).toEqual([42, 66, 120]);
  });

  it('fails closed when spatial-order metadata does not match the model responses', () => {
    const incompatible = structuredClone(manifestJson) as unknown as Record<string, unknown>;
    incompatible.responseOrdering = 'xyz-c-order';
    expect(() => validateBrainMapManifest(
      incompatible,
      'https://infoviz.example/data/brain/nsd-subj01/manifest.json',
    )).toThrow(BrainMapAssetError);
  });

  it('verifies fetched bytes before a renderer can parse them', async () => {
    const fetcher = async () => new Response(new TextEncoder().encode('abc'), { status: 200 });
    const expected = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

    await expect(fetchVerifiedBrainAsset(
      'https://infoviz.example/inflated.glb',
      expected,
      'test surface',
      fetcher,
    )).resolves.toHaveProperty('byteLength', 3);
    await expect(fetchVerifiedBrainAsset(
      'https://infoviz.example/inflated.glb',
      '0'.repeat(64),
      'test surface',
      fetcher,
    )).rejects.toThrow(/integrity check/i);
  });
});
