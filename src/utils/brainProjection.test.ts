import { describe, expect, it } from 'vitest';
import type { BrainProjection } from '../types/brainMap';
import type { CanonicalRoi } from '../types/session';
import { divergentResponseColor, fixedSessionResponseExtent, projectHemisphereResponses } from './brainProjection';

function roiProjection() {
  return {
    responseCount: 1,
    mappedResponseCount: 1,
    volumeCoordinatesZyx: [[0, 0, 0] as [number, number, number]],
    volumeLinearIndicesXyz: [0],
    unmappedResponseIndices: [],
    hemispheres: ['lh' as const],
    mappings: {
      lh: { responseIndices: [0], vertexIndices: [0], weights: [1] },
      rh: { responseIndices: [], vertexIndices: [], weights: [] },
    },
  };
}

describe('brain response surface projection', () => {
  it('resolves ROI overlap by absolute projected response and reports the overlap', () => {
    const projection: BrainProjection = {
      schemaVersion: 1,
      subjectId: 's1',
      responseSpace: 'nsd_1000_subject1_floc_t7_v1',
      fLocThreshold: 7,
      fLocThresholdOperator: '>',
      responseOrdering: 'zyx-c-order',
      mapper: 'linenn',
      normalizationPolicy: 'per-surface-vertex',
      volumeShapeXyz: [83, 104, 81],
      rois: { ffa: roiProjection(), ppa: roiProjection(), eba: roiProjection() },
    };
    const roiResponses: Record<CanonicalRoi, number[]> = {
      ffa: [0.25],
      ppa: [-0.8],
      eba: [0.5],
    };
    const result = projectHemisphereResponses({
      projection,
      roiResponses,
      enabledRois: new Set<CanonicalRoi>(['ffa', 'ppa', 'eba']),
      hemisphere: 'lh',
      vertexCount: 1,
    });

    expect(result.values[0]).toBeCloseTo(-0.8);
    expect(result.vertexData[0]?.dominantSource.roi).toBe('ppa');
    expect(result.vertexData[0]?.overlappingRois).toEqual(['ffa', 'ppa', 'eba']);
  });

  it('uses a shared zero-centered divergent color scale', () => {
    expect(divergentResponseColor(0, 2)).toEqual([242 / 255, 239 / 255, 232 / 255]);
    expect(divergentResponseColor(-2, 2)[2]).toBeGreaterThan(divergentResponseColor(2, 2)[2]);
    expect(divergentResponseColor(2, 2)[0]).toBeGreaterThan(divergentResponseColor(-2, 2)[0]);
  });

  it('locks the scale across stimuli and ROIs', () => {
    expect(fixedSessionResponseExtent([
      { stimulusId: 'a', roiResponses: { ffa: [0.2], ppa: [-0.4], eba: [0.1] } },
      { stimulusId: 'b', roiResponses: { ffa: [1.7], ppa: [-0.3], eba: [-2.1] } },
    ])).toBe(2.1);
  });
});
