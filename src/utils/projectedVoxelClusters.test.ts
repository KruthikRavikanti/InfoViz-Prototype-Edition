import { describe, expect, it } from 'vitest';
import brainManifestJson from '../../public/data/brain/nsd-subj01/manifest.json';
import brainProjectionJson from '../../public/data/brain/nsd-subj01/projection.json';
import atlasManifestJson from '../../public/data/voxel_selectivity_atlas/manifest.json';
import pointsText from '../../public/data/voxel_selectivity_atlas/clip_rn50/points.csv?raw';
import summaryJson from '../../public/data/voxel_selectivity_atlas/clip_rn50/summary.json';
import { validateBrainMapManifest, validateBrainProjection } from '../services/brainMapAssets';
import {
  parseSelectivityAtlasPoints, validateSelectivityAtlasManifest, validateSelectivityAtlasSummary,
} from '../services/selectivityAtlasAssets';
import type { BrainMapAssets, BrainProjectionRoi } from '../types/brainMap';
import type { SelectivityAtlasPoint } from '../types/selectivityAtlas';
import { CANONICAL_ROIS, type CanonicalRoi } from '../types/session';
import { projectVoxelClusters } from './projectedVoxelClusters';

const manifest = validateBrainMapManifest(brainManifestJson, 'https://example.test/manifest.json');
const assets: BrainMapAssets = { manifest, projection: validateBrainProjection(brainProjectionJson, manifest) };
const atlas = validateSelectivityAtlasManifest(atlasManifestJson);
const summary = validateSelectivityAtlasSummary(summaryJson, atlas, 'clip_rn50');
const points = parseSelectivityAtlasPoints(pointsText, atlas, summary);

function fixture() {
  const brain = structuredClone(assets);
  brain.manifest.surfaces.inflated!.hemispheres.lh.vertexCount = 5;
  const empty = (): BrainProjectionRoi => ({
    responseCount: 0, mappedResponseCount: 0, volumeCoordinatesZyx: [], volumeLinearIndicesXyz: [],
    unmappedResponseIndices: [], hemispheres: [],
    mappings: {
      lh: { responseIndices: [], vertexIndices: [], weights: [] },
      rh: { responseIndices: [], vertexIndices: [], weights: [] },
    },
  });
  brain.projection.rois = { ffa: empty(), ppa: empty(), eba: empty() };
  const source = (roi: CanonicalRoi, local: number, label: number, global: number): SelectivityAtlasPoint => ({
    roi, roiVoxelIndex: local, clusterLabel: label, voxelIndex: global,
    umapX: 9999, umapY: -200, umapZ: 10, distanceToCentroid: 0.1,
  });
  const fixturePoints = [source('ppa', 0, 1, 800), source('ppa', 1, 11, 801), source('ffa', 0, 11, 0)];
  brain.projection.rois.ppa.responseCount = 2;
  brain.projection.rois.ppa.mappings.lh = {
    responseIndices: [0, 1, 0, 1, 0, 1, 0],
    vertexIndices: [0, 0, 1, 1, 2, 2, 0],
    weights: [0.125, 0.5, 0.75, 0.25, 0.5, 0.5, 0.125],
  };
  brain.projection.rois.ffa.responseCount = 1;
  brain.projection.rois.ffa.mappings.lh = { responseIndices: [0], vertexIndices: [1], weights: [4] };
  return { assets: brain, points: fixturePoints, hemisphere: 'lh' as const, vertexCount: 5,
    enabledRois: new Set<CanonicalRoi>(['ppa']) };
}

describe('categorical voxel cluster surface projection', () => {
  it('sums source weights, resolves exact ties by label, and preserves unsupported vertices', () => {
    const result = projectVoxelClusters(fixture());
    expect([...result.labels]).toEqual([11, 1, 1, -1, -1]);
    expect([...result.support]).toEqual([0.75, 1, 1, 0, 0]);
    expect(result.clusterWeights.get(0)).toEqual(new Map([[1, 0.25], [11, 0.5]]));
    expect(result.contributors.get(0)?.map(({ point, weight }) => [point.roiVoxelIndex, weight]))
      .toEqual([[1, 0.5], [0, 0.25]]);
    expect(result.contributors.has(3)).toBe(false);
  });

  it('excludes disabled ROI sources even when they use a visible cluster label', () => {
    const input = fixture();
    const ppa = projectVoxelClusters(input);
    const all = projectVoxelClusters({ ...input, enabledRois: new Set(CANONICAL_ROIS) });
    expect(ppa.labels[1]).toBe(1);
    expect(all.labels[1]).toBe(11);
    expect(ppa.contributors.get(1)?.every(({ point }) => point.roi === 'ppa')).toBe(true);
    expect(all.contributors.get(1)?.[0].point.roi).toBe('ffa');
    const hidden = projectVoxelClusters({ ...input, enabledRois: new Set() });
    expect([...hidden.labels]).toEqual([-1, -1, -1, -1, -1]);
    expect(hidden.clusterWeights.size).toBe(0);
  });

  it('uses ROI-local identities regardless of embedding coordinates or point order', () => {
    const input = fixture();
    const original = projectVoxelClusters(input);
    const shuffled = projectVoxelClusters({ ...input, points: [...input.points].reverse().map((point) => ({
      ...point, voxelIndex: 9000, umapX: 0, umapY: 0, umapZ: 0,
    })) });
    expect(shuffled.labels).toEqual(original.labels);
    expect(shuffled.clusterWeights).toEqual(original.clusterWeights);
    expect(shuffled.contributors.get(2)?.map(({ point }) => point.roiVoxelIndex)).toEqual([0, 1]);
  });

  it('rejects missing or duplicate sources, invalid cluster labels and invalid mapper indices', () => {
    const input = fixture();
    expect(() => projectVoxelClusters({ ...input, points: input.points.slice(1) })).toThrow('incomplete');
    expect(() => projectVoxelClusters({ ...input, points: [...input.points, input.points[0]] })).toThrow('duplicate');
    expect(() => projectVoxelClusters({ ...input, points: input.points.map((point) => ({ ...point, clusterLabel: 14 })) }))
      .toThrow('invalid');
    input.assets.projection.rois.ppa.mappings.lh.vertexIndices[0] = 5;
    expect(() => projectVoxelClusters(input)).toThrow('invalid source');
    expect(() => projectVoxelClusters({ ...fixture(), vertexCount: 6 })).toThrow('mesh');
    const nonfinite = fixture();
    nonfinite.assets.projection.rois.ppa.mappings.lh.weights[0] = NaN;
    expect(() => projectVoxelClusters(nonfinite)).toThrow('weight');
  });

  it('retains all 170 audited Subject 1 PPA sources and labels exactly 1183 supported mesh vertices', () => {
    expect(points.filter((point) => point.roi === 'ppa' && point.clusterLabel === 1)).toHaveLength(100);
    expect(points.filter((point) => point.roi === 'ppa' && point.clusterLabel === 11)).toHaveLength(70);
    const observedSources = new Map<number, Set<number>>([[1, new Set()], [11, new Set()]]);
    for (const [hemisphere, c1, c11, nativeC1, nativeC11] of [
      ['lh', 235, 243, 36, 34], ['rh', 443, 262, 64, 36],
    ] as const) {
      const result = projectVoxelClusters({ assets, points, enabledRois: new Set(['ppa']), hemisphere,
        vertexCount: manifest.surfaces.inflated!.hemispheres[hemisphere].vertexCount });
      expect(result.labels.filter((label) => label === 1)).toHaveLength(c1);
      expect(result.labels.filter((label) => label === 11)).toHaveLength(c11);
      expect(result.labels.filter((label) => label >= 0)).toHaveLength(c1 + c11);
      const native = new Map<number, Set<number>>([[1, new Set()], [11, new Set()]]);
      for (const [vertex, contributors] of result.contributors) {
        expect(result.support[vertex]).toBeGreaterThan(0);
        for (const { point, weight } of contributors) {
          expect(point.roi).toBe('ppa');
          expect(weight).toBeGreaterThan(0);
          native.get(point.clusterLabel)!.add(point.roiVoxelIndex);
          observedSources.get(point.clusterLabel)!.add(point.roiVoxelIndex);
        }
      }
      expect(native.get(1)?.size).toBe(nativeC1);
      expect(native.get(11)?.size).toBe(nativeC11);
      expect(result.clusterWeights.size).toBe(c1 + c11);
    }
    expect(observedSources.get(1)?.size).toBe(100);
    expect(observedSources.get(11)?.size).toBe(70);
  });
});
