import type { EvidenceImage, ModelRoiColumn, Roi, SelectedHeatmapCell, WideCsvRow } from '../types/data';
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
    markerColor: stimulus.color,
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
  session: CortexComparisonSession,
  focusedStimulusId: string | null,
  imageLimit: number,
): SessionEvidenceView | null {
  if (selectedCell.model !== session.comparison.model || !isCanonicalRoi(selectedCell.roi)) {
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
): SessionClusterOverlay[] {
  return session.roiResults[roi].stimuli.map((result) => {
    const stimulus = session.stimuli.find((candidate) => candidate.id === result.stimulusId)!;
    return {
      stimulusId: stimulus.id,
      displayName: stimulus.displayName,
      imageUrl: stimulus.imageUrl,
      color: stimulus.color,
      x: result.voxelProjection.x,
      y: result.voxelProjection.y,
      clusterId: result.voxelProjection.clusterId,
      distance: result.voxelProjection.centroidDistance,
    };
  });
}

export function buildOptionalSessionOverlays(
  session: CortexComparisonSession,
  results: OptionalProjectionResults | undefined,
): SessionClusterOverlay[] {
  if (!results) return [];
  return results.stimuli.flatMap((result) => {
    if (!result.projection) return [];
    const stimulus = session.stimuli.find((candidate) => candidate.id === result.stimulusId);
    if (!stimulus) return [];
    return [{
      stimulusId: stimulus.id,
      displayName: stimulus.displayName,
      imageUrl: stimulus.imageUrl,
      color: stimulus.color,
      x: result.projection.x,
      y: result.projection.y,
      clusterId: result.projection.clusterId,
      distance: result.projection.centroidDistance,
    }];
  });
}
