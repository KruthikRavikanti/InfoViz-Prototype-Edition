import type {
  BrainHemisphere,
  BrainHemisphereProjection,
  BrainProjection,
  BrainSurfaceVertexDatum,
} from '../types/brainMap';
import { CANONICAL_ROIS, type BrainMapStimulusResult, type CanonicalRoi } from '../types/session';

type RoiAccumulator = {
  sums: Float64Array;
  weights: Float64Array;
  dominantContribution: Float64Array;
  dominantResponseIndex: Int32Array;
  dominantMappingWeight: Float64Array;
};

const COLOR_NEGATIVE: [number, number, number] = [42 / 255, 92 / 255, 158 / 255];
const COLOR_NEUTRAL: [number, number, number] = [242 / 255, 239 / 255, 232 / 255];
const COLOR_POSITIVE: [number, number, number] = [180 / 255, 48 / 255, 42 / 255];

function blend(
  from: [number, number, number],
  to: [number, number, number],
  amount: number,
): [number, number, number] {
  return [
    from[0] + (to[0] - from[0]) * amount,
    from[1] + (to[1] - from[1]) * amount,
    from[2] + (to[2] - from[2]) * amount,
  ];
}

export function divergentResponseColor(value: number, maxAbs: number): [number, number, number] {
  const denominator = maxAbs > 0 && Number.isFinite(maxAbs) ? maxAbs : 1;
  const normalized = Math.max(-1, Math.min(1, value / denominator));
  return normalized < 0
    ? blend(COLOR_NEUTRAL, COLOR_NEGATIVE, -normalized)
    : blend(COLOR_NEUTRAL, COLOR_POSITIVE, normalized);
}

export function commonResponseExtent(
  roiResponses: Record<CanonicalRoi, number[]>,
  enabledRois: ReadonlySet<CanonicalRoi>,
): number {
  let maxAbs = 0;
  for (const roi of CANONICAL_ROIS) {
    if (!enabledRois.has(roi)) continue;
    for (const response of roiResponses[roi]) maxAbs = Math.max(maxAbs, Math.abs(response));
  }
  return maxAbs || 1;
}

export function fixedSessionResponseExtent(stimuli: BrainMapStimulusResult[]): number {
  let maxAbs = 0;
  for (const stimulus of stimuli) {
    for (const roi of CANONICAL_ROIS) {
      for (const response of stimulus.roiResponses[roi]) maxAbs = Math.max(maxAbs, Math.abs(response));
    }
  }
  return maxAbs || 1;
}

function createAccumulator(vertexCount: number): RoiAccumulator {
  const dominantResponseIndex = new Int32Array(vertexCount);
  dominantResponseIndex.fill(-1);
  return {
    sums: new Float64Array(vertexCount),
    weights: new Float64Array(vertexCount),
    dominantContribution: new Float64Array(vertexCount),
    dominantResponseIndex,
    dominantMappingWeight: new Float64Array(vertexCount),
  };
}

/**
 * Project model responses onto one cortical hemisphere.
 *
 * Each ROI is first averaged independently at every reached surface vertex.
 * If enabled ROIs overlap, the ROI with the largest absolute projected value
 * owns the rendered color. CANONICAL_ROIS order breaks exact ties, making the
 * result stable regardless of object-key or mapping iteration order.
 */
export function projectHemisphereResponses({
  projection,
  roiResponses,
  enabledRois,
  hemisphere,
  vertexCount,
}: {
  projection: BrainProjection;
  roiResponses: Record<CanonicalRoi, number[]>;
  enabledRois: ReadonlySet<CanonicalRoi>;
  hemisphere: BrainHemisphere;
  vertexCount: number;
}): BrainHemisphereProjection {
  const accumulators = new Map<CanonicalRoi, RoiAccumulator>();
  for (const roi of CANONICAL_ROIS) {
    if (!enabledRois.has(roi)) continue;
    const accumulator = createAccumulator(vertexCount);
    const mapping = projection.rois[roi].mappings[hemisphere];
    for (let index = 0; index < mapping.responseIndices.length; index += 1) {
      const responseIndex = mapping.responseIndices[index];
      const vertexIndex = mapping.vertexIndices[index];
      const weight = mapping.weights[index];
      const response = roiResponses[roi][responseIndex];
      accumulator.sums[vertexIndex] += response * weight;
      accumulator.weights[vertexIndex] += weight;

      const contribution = Math.abs(response * weight);
      const currentContribution = accumulator.dominantContribution[vertexIndex];
      const currentIndex = accumulator.dominantResponseIndex[vertexIndex];
      if (
        contribution > currentContribution
        || (contribution === currentContribution && (currentIndex < 0 || responseIndex < currentIndex))
      ) {
        accumulator.dominantContribution[vertexIndex] = contribution;
        accumulator.dominantResponseIndex[vertexIndex] = responseIndex;
        accumulator.dominantMappingWeight[vertexIndex] = weight;
      }
    }
    accumulators.set(roi, accumulator);
  }

  const values = new Float32Array(vertexCount);
  const mapped = new Uint8Array(vertexCount);
  const vertexData: Array<BrainSurfaceVertexDatum | null> = Array.from({ length: vertexCount }, () => null);
  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    const overlappingRois: CanonicalRoi[] = [];
    let winner: CanonicalRoi | null = null;
    let winnerValue = 0;
    for (const roi of CANONICAL_ROIS) {
      const accumulator = accumulators.get(roi);
      if (!accumulator || accumulator.weights[vertexIndex] <= 0) continue;
      const value = accumulator.sums[vertexIndex] / accumulator.weights[vertexIndex];
      overlappingRois.push(roi);
      if (winner === null || Math.abs(value) > Math.abs(winnerValue)) {
        winner = roi;
        winnerValue = value;
      }
    }
    if (!winner) continue;

    const winnerAccumulator = accumulators.get(winner)!;
    const responseIndex = winnerAccumulator.dominantResponseIndex[vertexIndex];
    mapped[vertexIndex] = 1;
    values[vertexIndex] = winnerValue;
    vertexData[vertexIndex] = {
      surfaceResponse: winnerValue,
      overlappingRois,
      dominantSource: {
        roi: winner,
        responseIndex,
        response: roiResponses[winner][responseIndex],
        mappingWeight: winnerAccumulator.dominantMappingWeight[vertexIndex],
      },
    };
  }
  return { values, mapped, vertexData };
}
