import { describe, expect, it } from 'vitest';
import brainManifestJson from '../../public/data/brain/nsd-subj01/manifest.json';
import brainProjectionJson from '../../public/data/brain/nsd-subj01/projection.json';
import atlasManifestJson from '../../public/data/voxel_selectivity_atlas/manifest.json';
import clipPointsText from '../../public/data/voxel_selectivity_atlas/clip_rn50/points.csv?raw';
import clipSummaryJson from '../../public/data/voxel_selectivity_atlas/clip_rn50/summary.json';
import { validateBrainMapManifest, validateBrainProjection } from '../services/brainMapAssets';
import {
  parseSelectivityAtlasPoints,
  validateSelectivityAtlasManifest,
  validateSelectivityAtlasSummary,
} from '../services/selectivityAtlasAssets';
import type { BrainMapAssets, BrainProjectionRoi } from '../types/brainMap';
import type { SelectivityAtlasAssets, SelectivityAtlasPoint } from '../types/selectivityAtlas';
import { CANONICAL_ROIS, type CanonicalRoi } from '../types/session';
import {
  buildSelectivityBrainAnchors,
  projectSelectivityHemisphere,
  validateSelectivityBrainCompatibility,
} from './selectivityBrainProjection';

const pointsByPath = import.meta.glob('../../public/data/voxel_selectivity_atlas/*/points.csv', {
  query: '?raw', import: 'default', eager: true,
}) as Record<string, string>;
const summariesByPath = import.meta.glob('../../public/data/voxel_selectivity_atlas/*/summary.json', {
  query: '?raw', import: 'default', eager: true,
}) as Record<string, string>;

const brainManifest = validateBrainMapManifest(brainManifestJson, 'https://example.test/data/brain/nsd-subj01/manifest.json');
const brain: BrainMapAssets = {
  manifest: brainManifest,
  projection: validateBrainProjection(brainProjectionJson, brainManifest),
};
const manifest = validateSelectivityAtlasManifest(atlasManifestJson);
const summary = validateSelectivityAtlasSummary(clipSummaryJson, manifest, 'clip_rn50');
const points = parseSelectivityAtlasPoints(clipPointsText, manifest, summary);
const atlas: SelectivityAtlasAssets = { manifest, summary, points };
const allRois = new Set(CANONICAL_ROIS);

function identity(point: SelectivityAtlasPoint): string {
  return `${point.roi}:${point.roiVoxelIndex}`;
}

function simpleRoi(count: number, candidates: number[][]): BrainProjectionRoi {
  const responseIndices: number[] = [];
  const vertexIndices: number[] = [];
  const weights: number[] = [];
  candidates.forEach((vertices, source) => {
    vertices.forEach((vertex, index) => {
      responseIndices.push(source);
      vertexIndices.push(vertex);
      weights.push(1 - index / 10);
    });
  });
  const unmappedResponseIndices = candidates.flatMap((vertices, index) => vertices.length ? [] : [index]);
  return {
    responseCount: count,
    mappedResponseCount: count - unmappedResponseIndices.length,
    volumeCoordinatesZyx: Array.from({ length: count }, (_, index) => [0, 0, index]),
    volumeLinearIndicesXyz: Array.from({ length: count }, (_, index) => index),
    unmappedResponseIndices,
    hemispheres: candidates.map((vertices) => vertices.length ? 'lh' : 'unmapped'),
    mappings: {
      lh: { responseIndices, vertexIndices, weights },
      rh: { responseIndices: [], vertexIndices: [], weights: [] },
    },
  };
}

function simpleFixture(candidates: number[][]): { assets: BrainMapAssets; points: SelectivityAtlasPoint[] } {
  const assets = structuredClone(brain);
  assets.projection.rois = {
    ffa: simpleRoi(candidates.length, candidates),
    ppa: simpleRoi(0, []),
    eba: simpleRoi(0, []),
  };
  return {
    assets,
    points: candidates.map((_, index) => ({
      voxelIndex: index,
      roi: 'ffa',
      roiVoxelIndex: index,
      clusterLabel: index % 14,
      umapX: -1000 - index,
      umapY: 5000,
      umapZ: 3000,
      distanceToCentroid: 0.2,
    })),
  };
}

describe('selectivity atlas anatomical source projection', () => {
  it('joins every reviewed model through ROI-local Subject 1 identities, never global ROI offsets', () => {
    expect(manifest.supportedModels).toHaveLength(126);
    expect(manifest.roiOrder).toEqual(['ffa', 'eba', 'ppa']);
    expect(CANONICAL_ROIS).toEqual(['ffa', 'ppa', 'eba']);
    for (const model of manifest.supportedModels) {
      const base = `../../public/data/voxel_selectivity_atlas/${model}`;
      const modelSummary = validateSelectivityAtlasSummary(JSON.parse(summariesByPath[`${base}/summary.json`]), manifest, model);
      const modelPoints = parseSelectivityAtlasPoints(pointsByPath[`${base}/points.csv`], manifest, modelSummary);
      const modelAtlas = { manifest, summary: modelSummary, points: modelPoints };
      validateSelectivityBrainCompatibility(modelAtlas, brain);
      const anchors = buildSelectivityBrainAnchors(brain, modelPoints);
      expect(modelPoints).toHaveLength(844);
      expect(anchors).toHaveLength(841);
      expect(new Set(anchors.map((anchor) => identity(anchor.point))).size).toBe(841);
      expect(new Set(anchors.map((anchor) => `${anchor.hemisphere}:${anchor.vertexIndex}`)).size).toBe(841);
      const sourcePoints = new Map(modelPoints.map((point) => [identity(point), point]));
      for (const anchor of anchors) {
        expect(anchor.point).toBe(sourcePoints.get(identity(anchor.point)));
      }
    }
  });

  it('places all mapped voxels at supported vertices, retaining discrete cluster assignments and native coordinates', () => {
    const anchors = buildSelectivityBrainAnchors(brain, points);
    expect(anchors.filter((anchor) => anchor.hemisphere === 'lh')).toHaveLength(299);
    expect(anchors.filter((anchor) => anchor.hemisphere === 'rh')).toHaveLength(542);
    for (const roi of CANONICAL_ROIS) {
      const native = brain.projection.rois[roi];
      expect(anchors.filter((anchor) => anchor.point.roi === roi)).toHaveLength(native.mappedResponseCount);
      for (const anchor of anchors.filter((entry) => entry.point.roi === roi)) {
        const mapping = native.mappings[anchor.hemisphere];
        const edge = mapping.responseIndices.findIndex((source, index) =>
          source === anchor.point.roiVoxelIndex && mapping.vertexIndices[index] === anchor.vertexIndex);
        expect(edge).toBeGreaterThanOrEqual(0);
        expect(anchor.mappingWeight).toBeCloseTo(mapping.weights[edge]);
        expect(anchor.point.clusterLabel).toBe(points[manifest.rois[roi].offset + anchor.point.roiVoxelIndex].clusterLabel);
        expect(native.volumeCoordinatesZyx[anchor.point.roiVoxelIndex]).toHaveLength(3);
      }
    }
    expect(anchors.find((anchor) => identity(anchor.point) === 'eba:11')).toMatchObject({ hemisphere: 'lh', vertexIndex: 27310 });
    expect(anchors.find((anchor) => identity(anchor.point) === 'eba:109')).toMatchObject({ hemisphere: 'lh', vertexIndex: 28707 });
    expect(anchors.find((anchor) => identity(anchor.point) === 'eba:338')).toMatchObject({ hemisphere: 'rh', vertexIndex: 31403 });
  });

  it('keeps unmapped EBA voxels in the atlas but does not invent brain locations for them', () => {
    const anchors = buildSelectivityBrainAnchors(brain, points);
    const placed = new Set(anchors.map((anchor) => identity(anchor.point)));
    expect(points.filter((point) => !placed.has(identity(point))).map((point) => ({
      globalIndex: point.voxelIndex, roi: point.roi, localIndex: point.roiVoxelIndex,
    }))).toEqual([
      { globalIndex: 198, roi: 'eba', localIndex: 42 },
      { globalIndex: 222, roi: 'eba', localIndex: 66 },
      { globalIndex: 276, roi: 'eba', localIndex: 120 },
    ]);
  });

  it('keeps anatomical markers stable under shuffled embedding points or changed UMAP coordinates', () => {
    const original = buildSelectivityBrainAnchors(brain, points);
    const shuffled = [...points].reverse().map((point) => ({ ...point, umapX: -9999, umapY: 9999, umapZ: 0 }));
    const reordered = buildSelectivityBrainAnchors(brain, shuffled);
    expect(reordered.map(({ point, ...anchor }) => ({ identity: identity(point), cluster: point.clusterLabel, ...anchor })))
      .toEqual(original.map(({ point, ...anchor }) => ({ identity: identity(point), cluster: point.clusterLabel, ...anchor })));
  });

  it('filters ROI visibility after assigning anchors, without moving remaining voxels or reclustering', () => {
    for (const hemisphere of ['lh', 'rh'] as const) {
      const vertexCount = brain.manifest.surfaces.inflated!.hemispheres[hemisphere].vertexCount;
      const all = projectSelectivityHemisphere({ assets: brain, points, enabledRois: allRois, hemisphere, vertexCount });
      const filtered = projectSelectivityHemisphere({
        assets: brain, points, enabledRois: new Set<CanonicalRoi>(['ffa', 'ppa']), hemisphere, vertexCount,
      });
      expect(filtered.voxelAnchors).toEqual(all.voxelAnchors.filter((anchor) => anchor.point.roi !== 'eba'));
      expect(filtered.vertexData.filter(Boolean)).toHaveLength(hemisphere === 'lh' ? 129 : 197);
      for (const anchor of filtered.voxelAnchors) expect(filtered.vertexData[anchor.vertexIndex]).toBe(anchor);
      const hidden = projectSelectivityHemisphere({ assets: brain, points, enabledRois: new Set(), hemisphere, vertexCount });
      expect(hidden.voxelAnchors).toHaveLength(0);
      expect(hidden.vertexData.some(Boolean)).toBe(false);
    }
  });

  it('allocates single-candidate sources first rather than silently merging co-located source identities', () => {
    const fixture = simpleFixture([[3, 4], [3]]);
    const anchors = buildSelectivityBrainAnchors(fixture.assets, fixture.points);
    expect(anchors.map((anchor) => anchor.vertexIndex)).toEqual([4, 3]);
    expect(anchors.map((anchor) => anchor.point.clusterLabel)).toEqual([0, 1]);
    expect(anchors[1].sourceCount).toBe(2);
  });

  it('uses a deterministic augmenting path when constrained-first greedy placement still needs an alternative', () => {
    const fixture = simpleFixture([[0, 1], [0, 2], [0, 2]]);
    const anchors = buildSelectivityBrainAnchors(fixture.assets, fixture.points);
    expect(anchors).toHaveLength(3);
    expect(new Set(anchors.map((anchor) => anchor.vertexIndex)).size).toBe(3);
    for (const anchor of anchors) {
      expect(fixture.assets.projection.rois.ffa.mappings.lh.vertexIndices.some((vertex, index) =>
        vertex === anchor.vertexIndex && fixture.assets.projection.rois.ffa.mappings.lh.responseIndices[index] === anchor.point.roiVoxelIndex))
        .toBe(true);
    }
    expect(buildSelectivityBrainAnchors(fixture.assets, [...fixture.points].reverse())).toEqual(anchors);
  });

  it('fails closed when no distinct supported representative can exist', () => {
    const fixture = simpleFixture([[0], [0]]);
    expect(() => buildSelectivityBrainAnchors(fixture.assets, fixture.points)).toThrow(/distinct supported cortical markers/i);
  });

  it('breaks equal-weight ties deterministically by hemisphere and vertex index', () => {
    const fixture = simpleFixture([[4, 2]]);
    fixture.assets.projection.rois.ffa.mappings.lh.weights = [1, 1];
    fixture.assets.projection.rois.ffa.mappings.rh = { responseIndices: [0], vertexIndices: [0], weights: [1] };
    const anchor = buildSelectivityBrainAnchors(fixture.assets, fixture.points)[0];
    expect(anchor).toMatchObject({ hemisphere: 'lh', vertexIndex: 2, mappingWeight: 1 });
  });

  it('rejects a subject response-space or ROI-size mismatch before anatomy is shown', () => {
    const differentSpace = structuredClone(atlas);
    Object.assign(differentSpace.manifest, { responseSpace: 'different_subject' });
    expect(() => validateSelectivityBrainCompatibility(differentSpace, brain)).toThrow(/different subject response spaces/i);
    const differentCount = structuredClone(atlas);
    differentCount.manifest.rois.eba.count = 519;
    expect(() => validateSelectivityBrainCompatibility(differentCount, brain)).toThrow(/EBA atlas voxel ordering/i);
  });

  it('rejects incomplete, duplicate, out-of-range, or invalid categorical voxel identities', () => {
    expect(() => buildSelectivityBrainAnchors(brain, points.slice(1))).toThrow(/incomplete/i);
    expect(() => buildSelectivityBrainAnchors(brain, [...points, points[0]])).toThrow(/duplicate/i);
    expect(() => buildSelectivityBrainAnchors(brain, [{ ...points[0], roiVoxelIndex: -1 }, ...points.slice(1)])).toThrow(/invalid/i);
    expect(() => buildSelectivityBrainAnchors(brain, [{ ...points[0], clusterLabel: 14 }, ...points.slice(1)])).toThrow(/invalid/i);
  });

  it('rejects invalid mapper edges and mesh vertex-count mismatches', () => {
    const badWeight = simpleFixture([[0]]);
    badWeight.assets.projection.rois.ffa.mappings.lh.weights[0] = 0;
    expect(() => buildSelectivityBrainAnchors(badWeight.assets, badWeight.points)).toThrow(/invalid source or surface vertex/i);
    const badVertex = simpleFixture([[0]]);
    badVertex.assets.projection.rois.ffa.mappings.lh.vertexIndices[0] = 227021;
    expect(() => buildSelectivityBrainAnchors(badVertex.assets, badVertex.points)).toThrow(/invalid source or surface vertex/i);
    const badArrays = simpleFixture([[0]]);
    badArrays.assets.projection.rois.ffa.mappings.lh.weights = [];
    expect(() => buildSelectivityBrainAnchors(badArrays.assets, badArrays.points)).toThrow(/mapper arrays are incompatible/i);
    expect(() => projectSelectivityHemisphere({
      assets: brain, points, enabledRois: allRois, hemisphere: 'lh', vertexCount: 1,
    })).toThrow(/outside the cortical mesh/i);
  });
});
