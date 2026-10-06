import type { ClusterPoint, ClusterView, ProjectionAxisAlignment } from '../types/data';

type Coordinates = { x: number; y: number };

// Independent PCA computations may reverse an axis or differ by rounding noise.
// Never fit a translation, rotation, scale, or t-SNE transform to make them agree.
const ABSOLUTE_TOLERANCE = 1e-6;
const RELATIVE_TOLERANCE = 1e-7;

function approximatelyEqual(a: number, b: number): boolean {
  return Math.abs(a - b) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.max(Math.abs(a), Math.abs(b));
}

function displayedPcaCoordinates(point: ClusterPoint, view: ClusterView): Coordinates | null {
  const { pcaX, pcaY } = point;
  if (pcaX === null || pcaY === null || !Number.isFinite(pcaX) || !Number.isFinite(pcaY)) {
    return null;
  }

  if (view.coordinateMode === 'pca') {
    return { x: pcaX, y: pcaY };
  }

  if (view.coordinateMode !== 'plot' || point.plotX === null || point.plotY === null
    || !Number.isFinite(point.plotX) || !Number.isFinite(point.plotY)
    || !approximatelyEqual(point.plotX, pcaX) || !approximatelyEqual(point.plotY, pcaY)) {
    return null;
  }

  return { x: point.plotX, y: point.plotY };
}

function completeCoordinates(view: ClusterView): Map<string, Coordinates> | null {
  // A same-sized subset must not pass when the artifact declares a larger reference set.
  const expectedCount = view.summary?.nImages;
  if (expectedCount === null || expectedCount === undefined
    || !Number.isInteger(expectedCount) || expectedCount !== view.points.length) {
    return null;
  }

  if (view.points.length < 3) {
    return null;
  }

  const coordinates = new Map<string, Coordinates>();
  for (const point of view.points) {
    const value = displayedPcaCoordinates(point, view);
    if (!point.imageName || coordinates.has(point.imageName) || !value) {
      return null;
    }
    coordinates.set(point.imageName, value);
  }
  return coordinates;
}

function inferAxisSign(pairs: Array<{ source: Coordinates; target: Coordinates }>, axis: 'x' | 'y'): 1 | -1 | null {
  const candidates = ([1, -1] as const).filter((sign) =>
    pairs.every(({ source, target }) => approximatelyEqual(source[axis] * sign, target[axis])),
  );
  // An all-zero/near-zero axis cannot establish a reliable orientation.
  return candidates.length === 1 ? candidates[0] : null;
}

/**
 * Validate a sign-only mapping using every matched reference image's displayed
 * PCA coordinates. The saved upload projection uses sourceView's axes; the
 * returned signs place uploads on targetView without moving any reference dot.
 * Missing, duplicate, nonfinite, or incompatible reference geometry fails closed.
 */
export function inferClusterProjectionAlignment(
  sourceView: ClusterView | null,
  targetView: ClusterView | null,
): ProjectionAxisAlignment | null {
  if (!sourceView || !targetView || sourceView.points.length !== targetView.points.length) {
    return null;
  }

  const sourceCoordinates = completeCoordinates(sourceView);
  const targetCoordinates = completeCoordinates(targetView);
  if (!sourceCoordinates || !targetCoordinates) {
    return null;
  }

  const pairs: Array<{ source: Coordinates; target: Coordinates }> = [];
  for (const [imageName, source] of sourceCoordinates) {
    const target = targetCoordinates.get(imageName);
    if (!target) {
      return null;
    }
    pairs.push({ source, target });
  }

  const xSign = inferAxisSign(pairs, 'x');
  const ySign = inferAxisSign(pairs, 'y');
  return xSign !== null && ySign !== null ? { xSign, ySign } : null;
}
