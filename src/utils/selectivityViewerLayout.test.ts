import { describe, expect, it } from 'vitest';
import { Euler, PerspectiveCamera, Quaternion, Vector3 } from 'three';
import {
  orbitSelectivityCamera,
  resizedSelectivityCameraDistance,
  selectivityBoundsCenter,
  selectivityFrameDistance,
  selectivityNavigationIndex,
  selectivityOrientationAxes,
  SELECTIVITY_DEFAULT_CAMERA_DIRECTION,
  type SelectivityPosition,
} from './selectivityViewerLayout';

const translatedPoints: SelectivityPosition[] = [[10, 3, 8], [22, 9, 16], [18, 5, 12]];

describe('selectivity viewer framing', () => {
  it('frames the raw translated coordinates without changing their scientific axes', () => {
    const center = selectivityBoundsCenter(translatedPoints);
    expect(center).toEqual([16, 6, 12]);
    expect(translatedPoints[0]).toEqual([10, 3, 8]);
    expect(selectivityFrameDistance(translatedPoints, center, 1)).toBeGreaterThan(2);
    expect(selectivityFrameDistance(translatedPoints, center, 0.4)).toBeGreaterThan(selectivityFrameDistance(translatedPoints, center, 1.6));
  });

  it('preserves relative zoom over viewport aspect changes', () => {
    const center = selectivityBoundsCenter(translatedPoints);
    const oldFit = selectivityFrameDistance(translatedPoints, center, 1.6);
    const newFit = selectivityFrameDistance(translatedPoints, center, 0.4);
    const resized = resizedSelectivityCameraDistance(oldFit * 0.7, oldFit, newFit);
    expect(resized / newFit).toBeCloseTo(0.7);
    expect(resizedSelectivityCameraDistance(resized, newFit, oldFit)).toBeCloseTo(oldFit * 0.7);
  });

  it('fits every full-model point inside portrait and landscape perspective views', () => {
    const center = selectivityBoundsCenter(translatedPoints);
    for (const aspect of [0.4, 1.6]) {
      const distance = selectivityFrameDistance(translatedPoints, center, aspect);
      const camera = new PerspectiveCamera(38, aspect, 0.05, 2000);
      camera.position.copy(new Vector3(...center)).add(new Vector3(...SELECTIVITY_DEFAULT_CAMERA_DIRECTION).setLength(distance));
      camera.lookAt(new Vector3(...center));
      camera.updateMatrixWorld();
      for (const position of translatedPoints) {
        const projected = new Vector3(...position).project(camera);
        expect(Math.abs(projected.x)).toBeLessThanOrEqual(0.780001);
        expect(Math.abs(projected.y)).toBeLessThanOrEqual(0.780001);
        expect(projected.z).toBeGreaterThan(-1);
        expect(projected.z).toBeLessThan(1);
      }
    }
  });

  it('rejects invalid coordinates, camera bounds, and resize values', () => {
    expect(() => selectivityBoundsCenter([[NaN, 0, 0]])).toThrow();
    expect(() => selectivityFrameDistance(translatedPoints, [0, 0, 0], 0)).toThrow();
    expect(() => selectivityFrameDistance(translatedPoints, [0, 0, 0], 1, 180)).toThrow();
    expect(() => resizedSelectivityCameraDistance(10, 0, 10)).toThrow();
    expect(selectivityBoundsCenter([])).toEqual([0, 0, 0]);
  });
});

describe('selectivity camera orientation', () => {
  it('orbits around a nonzero data target rather than rotating the cloud around the origin', () => {
    const target: SelectivityPosition = [16, 6, 12];
    const position: SelectivityPosition = [16, 6, 22];
    const rotated = orbitSelectivityCamera(position, target, Math.PI / 2, 0);
    expect(rotated[0]).toBeCloseTo(26);
    expect(rotated[1]).toBeCloseTo(6);
    expect(rotated[2]).toBeCloseTo(12);
    expect(new Vector3(...rotated).distanceTo(new Vector3(...target))).toBeCloseTo(10);
    expect(target).toEqual([16, 6, 12]);
  });

  it('keeps the camera radius and avoids singular polar orientations', () => {
    const target: SelectivityPosition = [5, 6, 7];
    const rotated = orbitSelectivityCamera([15, 6, 7], target, 0, -Math.PI);
    expect(rotated[1]).toBeGreaterThan(6);
    expect(new Vector3(...rotated).distanceTo(new Vector3(...target))).toBeCloseTo(10);
    expect(rotated[0]).not.toBe(5);
  });

  it('projects labeled world axes with the inverse camera quaternion', () => {
    const quaternion = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), Math.PI / 2);
    const axes = selectivityOrientationAxes(quaternion.toArray());
    const inverse = quaternion.clone().invert();
    [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)].forEach((axis, index) => {
      const expected = axis.applyQuaternion(inverse).toArray();
      axes[index].forEach((coordinate, component) => expect(coordinate).toBeCloseTo(expected[component]));
    });
    expect(selectivityOrientationAxes([0, 0, 0, 1])).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  it('keeps the triad correct for an oblique camera with compound rotations', () => {
    const quaternion = new Quaternion().setFromEuler(new Euler(0.4, -0.7, 0.2));
    const axes = selectivityOrientationAxes(quaternion.toArray());
    const inverse = quaternion.clone().invert();
    [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)].forEach((axis, index) => {
      const expected = axis.applyQuaternion(inverse).toArray();
      axes[index].forEach((coordinate, component) => expect(coordinate).toBeCloseTo(expected[component]));
      expect(Math.hypot(...axes[index])).toBeCloseTo(1);
    });
    expect(new Vector3(...axes[0]).dot(new Vector3(...axes[1]))).toBeCloseTo(0);
  });
});

describe('visible voxel keyboard navigation', () => {
  it('starts at a visible endpoint and wraps sequential navigation', () => {
    expect(selectivityNavigationIndex(5, null, ']')).toBe(0);
    expect(selectivityNavigationIndex(5, null, '[')).toBe(4);
    expect(selectivityNavigationIndex(5, 4, ']')).toBe(0);
    expect(selectivityNavigationIndex(5, 0, '[')).toBe(4);
    expect(selectivityNavigationIndex(5, 2, 'PageDown')).toBe(3);
    expect(selectivityNavigationIndex(5, 2, 'PageUp')).toBe(1);
  });

  it('supports first/last and safely handles a changed or empty ROI subset', () => {
    expect(selectivityNavigationIndex(5, 3, 'Home')).toBe(0);
    expect(selectivityNavigationIndex(5, 3, 'End')).toBe(4);
    expect(selectivityNavigationIndex(2, 7, ']')).toBe(0);
    expect(selectivityNavigationIndex(0, null, ']')).toBeNull();
    expect(selectivityNavigationIndex(5, 2, 'ArrowRight')).toBeNull();
  });
});
