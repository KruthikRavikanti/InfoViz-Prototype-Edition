import type { ClusteringMethod, EvidenceImage, ModelRoiColumn, ProjectionAxisAlignment, Roi, SelectedHeatmapCell, WideCsvRow } from '../types/data';
import type {
  CanonicalRoi,
  CortexComparisonSession,
  OptionalProjectionResults,
  ReferenceNeighbor,
  SessionClusterOverlay,
  SessionStimulus,
  StimulusRoiResult,
} from '../types/session';
import { CANONICAL_ROIS } from '../types/session';
import { findModelRoiColumn, rankImagesByColumn } from './evidence';
import { jointClusterMedoidName } from './jointClustering';
import { getSessionForModel } from './sessionModels';

export const UPLOADED_STIMULUS_MARKER_COLOR = '#1c1917';

export type SessionEvidenceView = {
  roi: CanonicalRoi;
  referenceImages: EvidenceImage[];
  uploadedImages: EvidenceImage[];
  mergedImages: EvidenceImage[];
  topImages: EvidenceImage[];
  bottomImages: EvidenceImage[];
  focusedResult: StimulusRoiResult;
  focusedStimulus: SessionStimulus;
  predecessor: EvidenceImage | null;
  successor: EvidenceImage | null;
  closestByMean: EvidenceImage[];
  closestInVoxelSpace: EvidenceImage[];
  referenceDomain: [number, number];
};

export function isCanonicalRoi(roi: Roi | string): roi is CanonicalRoi {
  return CANONICAL_ROIS.includes(roi as CanonicalRoi);
}

function neighborToEvidence(neighbor: ReferenceNeighbor): EvidenceImage {
  return {
    imageName: neighbor.imageName,
    imageUrl: neighbor.imageUrl,
    value: neighbor.value,
    valueLabel: neighbor.distance === undefined
      ? `#${neighbor.rank} · ${neighbor.value.toFixed(3)}`
      : `Distance ${neighbor.distance.toFixed(3)}`,
    rank: neighbor.rank,
    origin: 'reference',
    referenceRank: neighbor.rank,
  };
}

function uploadToEvidence(stimulus: SessionStimulus, result: StimulusRoiResult): EvidenceImage {
  return {
    imageName: stimulus.displayName,
    imageUrl: stimulus.imageUrl,
    value: result.mean,
    valueLabel: `Session #${result.sessionRank} · ${result.mean.toFixed(3)}`,
    rank: result.sessionRank,
    origin: 'uploaded',
    stimulusId: stimulus.id,
    markerColor: UPLOADED_STIMULUS_MARKER_COLOR,
    referenceRank: result.referenceRank,
    referencePercentile: result.referencePercentile,
    sessionRank: result.sessionRank,
    sessionCount: result.sessionCount,
  };
}

export function mergeReferenceAndSessionRanks(
  referenceImages: EvidenceImage[],
  uploadedImages: EvidenceImage[],
  sessionCount: number,
): EvidenceImage[] {
  const uploadsByRank = new Map(uploadedImages.map((image) => [image.sessionRank ?? image.rank, image]));
  const merged: EvidenceImage[] = [];
  let referenceIndex = 0;

  for (let rank = 1; rank <= sessionCount; rank += 1) {
    const upload = uploadsByRank.get(rank);
    if (upload) {
      merged.push({ ...upload, rank });
      continue;
    }
    const reference = referenceImages[referenceIndex];
    referenceIndex += 1;
    if (reference) {
      merged.push({
        ...reference,
        rank,
        origin: 'reference',
        referenceRank: reference.rank,
        sessionRank: rank,
        sessionCount,
        valueLabel: `Session #${rank} · ref #${reference.rank}`,
      });
    }
  }

  if (merged.length !== sessionCount || referenceIndex !== referenceImages.length) {
    throw new Error('Server session ranks could not be merged with the fixed reference ordering.');
  }
  return merged;
}

export function buildSessionEvidenceView(
  selectedCell: SelectedHeatmapCell,
  rows: WideCsvRow[],
  modelRoiColumns: ModelRoiColumn[],
  providedSession: CortexComparisonSession,
  focusedStimulusId: string | null,
  imageLimit: number,
): SessionEvidenceView | null {
  const session = getSessionForModel(providedSession, selectedCell.model);
  if (!session || !isCanonicalRoi(selectedCell.roi)) {
    return null;
  }
  const modelRoiColumn = findModelRoiColumn(modelRoiColumns, session.comparison.model, selectedCell.roi);
  if (!modelRoiColumn) return null;

  const referenceImages = rankImagesByColumn(rows, modelRoiColumn.columnName).map((image) => ({
    ...image,
    origin: 'reference' as const,
    referenceRank: image.rank,
  }));
  const roiResult = session.roiResults[selectedCell.roi];
  const uploadedImages = roiResult.stimuli.map((result) => {
    const stimulus = session.stimuli.find((candidate) => candidate.id === result.stimulusId);
    if (!stimulus) throw new Error(`Session result references unknown stimulus ${result.stimulusId}.`);
    return uploadToEvidence(stimulus, result);
  });
  const sessionCount = roiResult.stimuli[0]?.sessionCount ?? (185 + uploadedImages.length);
  const mergedImages = mergeReferenceAndSessionRanks(referenceImages, uploadedImages, sessionCount);
  const focusedResult = roiResult.stimuli.find((result) => result.stimulusId === focusedStimulusId)
    ?? roiResult.stimuli[0];
  const focusedStimulus = session.stimuli.find((stimulus) => stimulus.id === focusedResult.stimulusId)
    ?? session.stimuli[0];
  const values = referenceImages.map((image) => image.value);

  return {
    roi: selectedCell.roi,
    referenceImages,
    uploadedImages,
    mergedImages,
    topImages: mergedImages.slice(0, imageLimit),
    bottomImages: mergedImages.slice(-imageLimit).reverse(),
    focusedResult,
    focusedStimulus,
    predecessor: focusedResult.predecessor ? neighborToEvidence(focusedResult.predecessor) : null,
    successor: focusedResult.successor ? neighborToEvidence(focusedResult.successor) : null,
    closestByMean: focusedResult.closestByMean.map(neighborToEvidence),
    closestInVoxelSpace: focusedResult.closestInVoxelSpace.map(neighborToEvidence),
    referenceDomain: [Math.min(...values), Math.max(...values)],
  };
}

export function buildVoxelSessionOverlays(
  session: CortexComparisonSession,
  roi: CanonicalRoi,
  method: ClusteringMethod = 'kmeans_euclidean',
  alignment?: ProjectionAxisAlignment | null,
): SessionClusterOverlay[] {
  const hierarchical = method === 'hierarchical_correlation';
  // The stored upload PCA coordinates follow the k-means reference axes.
  // Hierarchical plots can flip those axes, so never draw an unchecked overlay.
  const axisAlignment = hierarchical ? alignment : { xSign: 1, ySign: 1 };
  if (!axisAlignment) return [];
  const jointHierarchy = session.roiResults[roi].jointHierarchy;
  if (hierarchical && !jointHierarchy) return [];
  const jointUploads = new Map(jointHierarchy?.members
    .filter((member) => member.kind === 'uploaded').map((member) => [member.id, member]));
  const stimuli = new Map(session.stimuli.map((stimulus) => [stimulus.id, stimulus]));

  return session.roiResults[roi].stimuli.flatMap((result) => {
    const stimulus = stimuli.get(result.stimulusId);
    if (!stimulus) return [];
    const member = jointUploads.get(result.stimulusId);
    if (hierarchical && !member) return [];
    return [{
      stimulusId: stimulus.id,
      displayName: stimulus.displayName,
      imageUrl: stimulus.imageUrl,
      color: UPLOADED_STIMULUS_MARKER_COLOR,
      x: result.voxelProjection.x * axisAlignment.xSign,
      y: result.voxelProjection.y * axisAlignment.ySign,
      clusterId: member && hierarchical ? member.clusterId : result.voxelProjection.clusterId,
      distance: member && hierarchical ? member.distanceToMedoid : result.voxelProjection.centroidDistance,
      assignmentMethod: hierarchical ? 'joint hierarchical clustering' as const : 'nearest-centroid assignment' as const,
      ...(hierarchical && jointHierarchy && member
        ? { medoidImageName: jointClusterMedoidName(jointHierarchy, member.clusterId, session) }
        : {}),
    }];
  });
}

export function buildOptionalSessionOverlays(
  session: CortexComparisonSession,
  results: OptionalProjectionResults | undefined,
  method: ClusteringMethod = 'kmeans_euclidean',
  alignment?: ProjectionAxisAlignment | null,
): SessionClusterOverlay[] {
  if (!results) return [];
  const hierarchical = method === 'hierarchical_correlation';
  const axisAlignment = hierarchical ? alignment : { xSign: 1, ySign: 1 };
  if (!axisAlignment) return [];
  const jointHierarchy = results.jointHierarchy;
  if (hierarchical && !jointHierarchy) return [];
  const jointUploads = new Map(jointHierarchy?.members
    .filter((member) => member.kind === 'uploaded').map((member) => [member.id, member]));
  const stimuli = new Map(session.stimuli.map((stimulus) => [stimulus.id, stimulus]));

  return results.stimuli.flatMap((result) => {
    if (!result.projection) return [];
    const member = jointUploads.get(result.stimulusId);
    if (hierarchical && !member) return [];
    const stimulus = stimuli.get(result.stimulusId);
    if (!stimulus) return [];
    return [{
      stimulusId: stimulus.id,
      displayName: stimulus.displayName,
      imageUrl: stimulus.imageUrl,
      color: UPLOADED_STIMULUS_MARKER_COLOR,
      x: result.projection.x * axisAlignment.xSign,
      y: result.projection.y * axisAlignment.ySign,
      clusterId: hierarchical && member ? member.clusterId : result.projection.clusterId,
      distance: hierarchical && member ? member.distanceToMedoid : result.projection.centroidDistance,
      assignmentMethod: hierarchical ? 'joint hierarchical clustering' as const : 'nearest-centroid assignment' as const,
      ...(hierarchical && jointHierarchy && member
        ? { medoidImageName: jointClusterMedoidName(jointHierarchy, member.clusterId, session) }
        : {}),
    }];
  });
}
