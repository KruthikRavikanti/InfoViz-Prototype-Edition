import { describe, expect, it } from 'vitest';
import { inflatedHemisphereTranslations } from './brainSurfaceLayout';

describe('inflated cortical surface layout', () => {
  it('separates independently inflated hemispheres without changing vertex data', () => {
    const left = { min: -48, max: 48 };
    const right = { min: -47, max: 49 };
    const translation = inflatedHemisphereTranslations(left, right, 8);
    const transformedLeft = { min: left.min + translation.lh, max: left.max + translation.lh };
    const transformedRight = { min: right.min + translation.rh, max: right.max + translation.rh };

    expect(transformedRight.min - transformedLeft.max).toBeCloseTo(8);
    expect(transformedLeft.max).toBeCloseTo(-4);
    expect(transformedRight.min).toBeCloseTo(4);
  });
});
