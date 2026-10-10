import type { BrainHemisphere, BrainMapAssets } from '../types/brainMap';
import type { SelectivityAtlasAssets, SelectivityAtlasPoint } from '../types/selectivityAtlas';
import { CANONICAL_ROIS, type CanonicalRoi } from '../types/session';

export type SelectivityBrainAnchor = {
  point: SelectivityAtlasPoint;
  hemisphere: BrainHemisphere;
  vertexIndex: number;
  mappingWeight: number;
  sourceCount: number;
};

type Candidate = Omit<SelectivityBrainAnchor, 'point' | 'sourceCount'>;
type Source = { point: SelectivityAtlasPoint; candidates: Candidate[] };
const HEMISPHERES: BrainHemisphere[] = ['lh', 'rh'];
const sourceKey = (point: SelectivityAtlasPoint) => `${point.roi}:${point.roiVoxelIndex}`;
const vertexKey = (candidate: Candidate) => `${candidate.hemisphere}:${candidate.vertexIndex}`;

export function validateSelectivityBrainCompatibility(atlas: SelectivityAtlasAssets, brain: BrainMapAssets): void {
  if (atlas.manifest.subject !== brain.manifest.subjectId
    || atlas.manifest.responseSpace !== brain.manifest.responseSpace
    || atlas.summary.subject !== brain.projection.subjectId
    || atlas.summary.responseSpace !== brain.projection.responseSpace) {
    throw new Error('The voxel atlas and cortical surface use different subject response spaces.');
  }
  for (const roi of CANONICAL_ROIS) {
    if (atlas.manifest.rois[roi].count !== brain.projection.rois[roi].responseCount
      || atlas.manifest.rois[roi].count !== brain.manifest.rois[roi].responseCount) {
      throw new Error(`${roi.toUpperCase()} atlas voxel ordering is incompatible with the cortical surface.`);
    }
  }
  // The atlas and brain files have DIFFERENT global ROI orders. Only ROI-local
  // response indices are anatomical identities; UMAP/global indices are not.
  validatePointIdentities(brain, atlas.points);
}

function validatePointIdentities(assets: BrainMapAssets, points: SelectivityAtlasPoint[]): Map<string, SelectivityAtlasPoint> {
  const bySource = new Map<string, SelectivityAtlasPoint>();
  for (const point of points) {
    const projectionRoi = assets.projection.rois[point.roi];
    if (!projectionRoi || !Number.isInteger(point.roiVoxelIndex) || point.roiVoxelIndex < 0
      || point.roiVoxelIndex >= projectionRoi.responseCount
      || !Number.isInteger(point.clusterLabel) || point.clusterLabel < 0 || point.clusterLabel >= 14
      || bySource.has(sourceKey(point))) {
      throw new Error('The voxel atlas has an invalid or duplicate anatomical source identity.');
    }
    bySource.set(sourceKey(point), point);
  }
  for (const roi of CANONICAL_ROIS) {
    for (let index = 0; index < assets.projection.rois[roi].responseCount; index += 1) {
      if (!bySource.has(`${roi}:${index}`)) throw new Error(`${roi.toUpperCase()} atlas source voxels are incomplete.`);
    }
  }
  return bySource;
}

/**
 * One discrete marker for each mapped source voxel, at a vertex explicitly
 * supported by the reviewed mapper. Constrained sources are allocated first;
 * augmenting paths resolve collisions without jittering or inventing locations.
 * Never average numeric cluster labels. Allocate before ROI filtering so a
 * visibility toggle cannot move any marker or change its cluster assignment.
 */
export function buildSelectivityBrainAnchors(assets: BrainMapAssets, points: SelectivityAtlasPoint[]): SelectivityBrainAnchor[] {
  const bySource = validatePointIdentities(assets, points);
  const candidates = new Map<string, Map<string, Candidate>>();
  const vertexSources = new Map<string, Set<string>>();
  for (const roi of CANONICAL_ROIS) {
    for (const hemisphere of HEMISPHERES) {
      const mapping = assets.projection.rois[roi].mappings[hemisphere];
      if (mapping.responseIndices.length !== mapping.vertexIndices.length
        || mapping.responseIndices.length !== mapping.weights.length) throw new Error('Cortical mapper arrays are incompatible.');
      const vertexCount = assets.manifest.surfaces.inflated?.hemispheres[hemisphere].vertexCount;
      for (let edge = 0; edge < mapping.responseIndices.length; edge += 1) {
        const responseIndex = mapping.responseIndices[edge];
        const vertexIndex = mapping.vertexIndices[edge];
        const weight = mapping.weights[edge];
        const key = `${roi}:${responseIndex}`;
        if (!bySource.has(key) || !Number.isInteger(vertexIndex) || vertexIndex < 0
          || vertexCount === undefined || vertexIndex >= vertexCount || !Number.isFinite(weight) || weight <= 0) {
          throw new Error('Cortical mapper contains an invalid source or surface vertex.');
        }
        const candidate = { hemisphere, vertexIndex, mappingWeight: weight };
        const vKey = vertexKey(candidate);
        let sourceCandidates = candidates.get(key);
        if (!sourceCandidates) { sourceCandidates = new Map(); candidates.set(key, sourceCandidates); }
        const previous = sourceCandidates.get(vKey);
        sourceCandidates.set(vKey, { ...candidate, mappingWeight: weight + (previous?.mappingWeight ?? 0) });
        let contributors = vertexSources.get(vKey);
        if (!contributors) { contributors = new Set(); vertexSources.set(vKey, contributors); }
        contributors.add(key);
      }
    }
  }
  const sources: Source[] = [...bySource.entries()].flatMap(([key, point]) => {
    const supported = [...candidates.get(key)?.values() ?? []].sort((a, b) =>
      b.mappingWeight - a.mappingWeight
      || HEMISPHERES.indexOf(a.hemisphere) - HEMISPHERES.indexOf(b.hemisphere)
      || a.vertexIndex - b.vertexIndex);
    return supported.length ? [{ point, candidates: supported }] : [];
  });
  sources.sort((a, b) => a.candidates.length - b.candidates.length
    || CANONICAL_ROIS.indexOf(a.point.roi) - CANONICAL_ROIS.indexOf(b.point.roi)
    || a.point.roiVoxelIndex - b.point.roiVoxelIndex);
  const owners = new Map<string, Source>();
  const assignments = new Map<string, Candidate>();
  function assign(source: Source, candidate: Candidate) {
    const key = sourceKey(source.point);
    const old = assignments.get(key);
    if (old) owners.delete(vertexKey(old));
    assignments.set(key, candidate);
    owners.set(vertexKey(candidate), source);
  }
  function place(source: Source, visitedSources: Set<string>): boolean {
    const key = sourceKey(source.point);
    if (visitedSources.has(key)) return false;
    visitedSources.add(key);
    // Prefer a free supported vertex before moving an existing source marker.
    for (const candidate of source.candidates) {
      if (!owners.has(vertexKey(candidate))) { assign(source, candidate); return true; }
    }
    for (const candidate of source.candidates) {
      const owner = owners.get(vertexKey(candidate))!;
      if (place(owner, visitedSources)) { assign(source, candidate); return true; }
    }
    return false;
  }
  for (const source of sources) {
    if (!place(source, new Set())) throw new Error('Distinct supported cortical markers could not be assigned to every mapped voxel.');
  }
  return sources.map(({ point }) => {
    const candidate = assignments.get(sourceKey(point))!;
    return { point, ...candidate, sourceCount: vertexSources.get(vertexKey(candidate))!.size };
  }).sort((a, b) => a.point.voxelIndex - b.point.voxelIndex);
}

export function projectSelectivityHemisphere({ assets, points, enabledRois, hemisphere, vertexCount }: {
  assets: BrainMapAssets;
  points: SelectivityAtlasPoint[];
  enabledRois: ReadonlySet<CanonicalRoi>;
  hemisphere: BrainHemisphere;
  vertexCount: number;
}): { voxelAnchors: SelectivityBrainAnchor[]; vertexData: Array<SelectivityBrainAnchor | null> } {
  const voxelAnchors = buildSelectivityBrainAnchors(assets, points)
    .filter((anchor) => anchor.hemisphere === hemisphere && enabledRois.has(anchor.point.roi));
  const vertexData: Array<SelectivityBrainAnchor | null> = Array.from({ length: vertexCount }, () => null);
  for (const anchor of voxelAnchors) {
    if (anchor.vertexIndex >= vertexCount) throw new Error('A voxel marker is outside the cortical mesh.');
    vertexData[anchor.vertexIndex] = anchor;
  }
  return { voxelAnchors, vertexData };
}
