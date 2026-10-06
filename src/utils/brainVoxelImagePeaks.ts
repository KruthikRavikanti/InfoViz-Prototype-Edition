import { CANONICAL_ROIS, SUBJECT_ONE_ROI_VOXEL_COUNTS, type CanonicalRoi } from '../types/session';
import type { BrainMapImage, BrainMapImageSource } from './brainMapSelection';

export type BrainVoxelImagePeak = {
  readonly image: BrainMapImage;
  /** Highest signed raw prediction, not its absolute magnitude. */
  readonly response: number;
  /** Number of images whose raw prediction equals this maximum exactly. */
  readonly tieCount: number;
};

export type BrainVoxelImagePeakIndex = {
  readonly status: 'ready' | 'empty' | 'invalid';
  readonly source: BrainMapImageSource;
  readonly candidateCount: number;
  readonly peaks: Readonly<Record<CanonicalRoi, readonly BrainVoxelImagePeak[]>>;
};

const EMPTY_PEAKS: BrainVoxelImagePeakIndex['peaks'] = Object.freeze({
  ffa: Object.freeze([]),
  ppa: Object.freeze([]),
  eba: Object.freeze([]),
});

/**
 * Cache one signed argmax per original Subject-1 ROI response column.
 *
 * The caller supplies the complete active image collection, not its search
 * results. Images from the other collection cannot participate. Predictions
 * remain raw: neither surface interpolation, absolute-value coloring nor
 * clustering normalization defines the winning image. Exact ties keep the
 * first image in the supplied order.
 *
 * A malformed candidate invalidates the complete lookup instead of silently
 * changing the comparison set by dropping it. Input images are never modified.
 */
export function buildBrainVoxelImagePeaks(
  images: readonly BrainMapImage[],
  source: BrainMapImageSource,
): BrainVoxelImagePeakIndex {
  const candidates = images.filter((image) => image?.source === source);
  if (candidates.length === 0) {
    return Object.freeze({ status: 'empty', source, candidateCount: 0, peaks: EMPTY_PEAKS });
  }

  for (const image of candidates) {
    for (const roi of CANONICAL_ROIS) {
      const responses = image.result?.roiResponses?.[roi];
      if (!Array.isArray(responses) || responses.length !== SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]) {
        return Object.freeze({ status: 'invalid', source, candidateCount: candidates.length, peaks: EMPTY_PEAKS });
      }
      // for...of also visits sparse array holes, unlike Array.prototype.every.
      for (const response of responses) {
        if (!Number.isFinite(response)) {
          return Object.freeze({ status: 'invalid', source, candidateCount: candidates.length, peaks: EMPTY_PEAKS });
        }
      }
    }
  }

  const peaks = Object.fromEntries(CANONICAL_ROIS.map((roi) => {
    const roiPeaks = Array.from({ length: SUBJECT_ONE_ROI_VOXEL_COUNTS[roi] }, (_, responseIndex) => {
      let image = candidates[0];
      let response = image.result.roiResponses[roi][responseIndex];
      let tieCount = 1;
      for (let candidateIndex = 1; candidateIndex < candidates.length; candidateIndex += 1) {
        const candidate = candidates[candidateIndex];
        const candidateResponse = candidate.result.roiResponses[roi][responseIndex];
        if (candidateResponse > response) {
          image = candidate;
          response = candidateResponse;
          tieCount = 1;
        } else if (candidateResponse === response) {
          tieCount += 1;
        }
      }
      return Object.freeze({ image, response, tieCount });
    });
    return [roi, Object.freeze(roiPeaks)];
  })) as Record<CanonicalRoi, readonly BrainVoxelImagePeak[]>;

  return Object.freeze({ status: 'ready', source, candidateCount: candidates.length, peaks: Object.freeze(peaks) });
}

/** Query the dominant source ROI/index already identified by the surface hover. */
export function getBrainVoxelImagePeak(
  index: BrainVoxelImagePeakIndex,
  roi: CanonicalRoi,
  responseIndex: number,
): BrainVoxelImagePeak | null {
  if (
    index.status !== 'ready'
    || !CANONICAL_ROIS.includes(roi)
    || !Number.isInteger(responseIndex)
    || responseIndex < 0
    || responseIndex >= SUBJECT_ONE_ROI_VOXEL_COUNTS[roi]
  ) return null;

  return index.peaks[roi][responseIndex] ?? null;
}
