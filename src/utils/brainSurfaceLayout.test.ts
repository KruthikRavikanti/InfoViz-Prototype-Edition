import { describe, expect, it } from 'vitest';
import { brainSurfaceCameraDistance, inflatedHemisphereTranslations } from './brainSurfaceLayout';

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

describe('responsive cortical surface camera framing', () => {
  it('fits both horizontal and vertical fields of view on narrow canvases', () => {
    const wideDistance = brainSurfaceCameraDistance(150, 1.8);
    expect(brainSurfaceCameraDistance(150, 1)).toBeCloseTo(wideDistance);
    expect(brainSurfaceCameraDistance(150, 0.6)).toBeGreaterThan(wideDistance);
    for (const aspect of [0.4, 0.7, 1, 1.8, 3]) {
      const distance = brainSurfaceCameraDistance(150, aspect);
      const verticalFov = 34 * Math.PI / 180;
      const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * aspect);
      expect(distance * Math.sin(verticalFov / 2)).toBeGreaterThan(150);
      expect(distance * Math.sin(horizontalFov / 2)).toBeGreaterThan(150);
    }
  });

  it('rejects invalid framing values instead of producing a clipped or nonfinite camera', () => {
    expect(() => brainSurfaceCameraDistance(0, 1)).toThrow('camera bounds');
    expect(() => brainSurfaceCameraDistance(150, 0)).toThrow('camera bounds');
    expect(() => brainSurfaceCameraDistance(150, 1, 180)).toThrow('camera bounds');
    expect(() => brainSurfaceCameraDistance(150, Number.NaN)).toThrow('camera bounds');
  });
});
