import { describe, expect, it } from 'vitest';
import { buildCategoricalSurfaceGeometry } from './categoricalSurfaceOverlay';

const positions = new Float32Array([0, 0, 0, 6, 0, 0, 0, 3, 0]);
const indices = new Uint32Array([0, 1, 2]);

function triangleAreas(coordinates: Float32Array): number[] {
  const areas: number[] = [];
  for (let i = 0; i < coordinates.length; i += 9) {
    const ab = [0, 1, 2].map((axis) => coordinates[i + 3 + axis] - coordinates[i + axis]);
    const ac = [0, 1, 2].map((axis) => coordinates[i + 6 + axis] - coordinates[i + axis]);
    const cross = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
    // Signed z area also verifies the source winding on this xy fixture.
    expect(cross[2]).toBeGreaterThan(0);
    areas.push(Math.hypot(...cross) / 2);
  }
  return areas;
}

describe('buildCategoricalSurfaceGeometry', () => {
  it('covers a fully supported face exactly, with constant categorical labels per triangle', () => {
    const result = buildCategoricalSurfaceGeometry({ positions, indices, labels: [1, 11, 1] });

    expect(result.triangleCount).toBe(6);
    expect(result.positions).toHaveLength(54);
    expect(result.clusterLabels).toHaveLength(18);
    expect(Array.from(result.sourceVertexIndices)).toEqual([0, 0, 1, 1, 2, 2]);
    expect(triangleAreas(result.positions).reduce((sum, area) => sum + area, 0)).toBeCloseTo(9);
    for (let triangle = 0; triangle < result.triangleCount; triangle += 1) {
      const labels = Array.from(result.clusterLabels.slice(triangle * 3, triangle * 3 + 3));
      const owner = result.sourceVertexIndices[triangle];
      expect(labels).toEqual(Array(3).fill([1, 11, 1][owner]));
      expect(triangleAreas(result.positions.slice(triangle * 9, triangle * 9 + 9))[0]).toBeCloseTo(1.5);
    }
  });

  it.each([0, 1, 2])('keeps only the supported third for source vertex %i, retaining unsupported holes', (supportedVertex) => {
    const labels = [-1, -1, -1];
    labels[supportedVertex] = 11;
    const result = buildCategoricalSurfaceGeometry({ positions, indices, labels });

    expect(result.triangleCount).toBe(2);
    expect(Array.from(result.sourceVertexIndices)).toEqual([supportedVertex, supportedVertex]);
    expect(Array.from(result.clusterLabels)).toEqual(Array(6).fill(11));
    expect(triangleAreas(result.positions).reduce((sum, area) => sum + area, 0)).toBeCloseTo(3);
    // The other original vertices never enter this cell: it stops at midpoints.
    const emitted = Array.from({ length: result.positions.length / 3 }, (_, vertex) => Array.from(result.positions.slice(vertex * 3, vertex * 3 + 3)));
    for (let vertex = 0; vertex < 3; vertex += 1) {
      if (vertex !== supportedVertex) expect(emitted).not.toContainEqual(Array.from(positions.slice(vertex * 3, vertex * 3 + 3)));
    }
  });

  it('emits no geometry for an unsupported face', () => {
    const result = buildCategoricalSurfaceGeometry({ positions, indices, labels: [-1, -1, -1] });
    expect(result.triangleCount).toBe(0);
    expect(result.positions).toHaveLength(0);
    expect(result.clusterLabels).toHaveLength(0);
    expect(result.sourceVertexIndices).toHaveLength(0);
    expect(result.normals).toBeUndefined();
  });

  it('interpolates geometry and normals within the original face without moving or relabeling input data', () => {
    const normals = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const labels = new Int16Array([11, -1, -1]);
    const original = { positions: positions.slice(), indices: indices.slice(), normals: normals.slice(), labels: labels.slice() };
    const result = buildCategoricalSurfaceGeometry({ positions, indices, normals, labels });

    expect(Array.from(result.positions.slice(0, 9))).toEqual([0, 0, 0, 3, 0, 0, 2, 1, 0]);
    expect(Array.from(result.normals!.slice(0, 6))).toEqual([1, 0, 0, 0.5, 0.5, 0]);
    for (const component of result.normals!.slice(6, 9)) expect(component).toBeCloseTo(1 / 3);
    expect(positions).toEqual(original.positions);
    expect(indices).toEqual(original.indices);
    expect(normals).toEqual(original.normals);
    expect(labels).toEqual(original.labels);
  });

  it('preserves the common boundary between cells across adjacent original triangles', () => {
    const meshPositions = new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 0]);
    const meshIndices = new Uint32Array([0, 1, 2, 1, 3, 2]);
    const result = buildCategoricalSurfaceGeometry({ positions: meshPositions, indices: meshIndices, labels: [-1, 1, 11, -1] });
    const vertices = Array.from({ length: result.positions.length / 3 }, (_, vertex) => Array.from(result.positions.slice(vertex * 3, vertex * 3 + 3)));

    expect(result.triangleCount).toBe(8);
    expect(vertices.filter((vertex) => vertex[0] === 1 && vertex[1] === 1 && vertex[2] === 0)).toHaveLength(4);
    expect(triangleAreas(result.positions).reduce((sum, area) => sum + area, 0)).toBeCloseTo(8 / 3);
  });

  it.each([
    { indices: [0, 1], error: 'complete triangles' },
    { indices: [0, 1, 3], error: 'invalid vertex index' },
    { indices: [0, -1, 2], error: 'invalid vertex index' },
    { indices: [0, 0.5, 2], error: 'invalid vertex index' },
    { indices: [0, 1, 1], error: 'repeated vertex indices' },
  ])('rejects malformed mesh indices: $indices', ({ indices: invalidIndices, error }) => {
    expect(() => buildCategoricalSurfaceGeometry({ positions, indices: invalidIndices, labels: [1, 11, -1] })).toThrow(error);
  });

  it('rejects geometric degeneracy even when indices are distinct', () => {
    expect(() => buildCategoricalSurfaceGeometry({ positions: [0, 0, 0, 1, 1, 1, 2, 2, 2], indices, labels: [1, 11, -1] })).toThrow('zero area');
  });

  it('rejects mismatched, nonfinite, and invalid categorical data', () => {
    expect(() => buildCategoricalSurfaceGeometry({ positions: [0, 0], indices: [], labels: [] })).toThrow('complete xyz vertices');
    expect(() => buildCategoricalSurfaceGeometry({ positions, indices, labels: [1] })).toThrow('vertex count');
    expect(() => buildCategoricalSurfaceGeometry({ positions, indices, labels: [1, 11, -1], normals: [0, 0, 1] })).toThrow('normal');
    expect(() => buildCategoricalSurfaceGeometry({ positions: [NaN, 0, 0], indices: [], labels: [-1] })).toThrow('finite');
    expect(() => buildCategoricalSurfaceGeometry({ positions, indices, labels: [1, 0.5, -1] })).toThrow('Int16');
    expect(() => buildCategoricalSurfaceGeometry({ positions, indices, labels: [1, 32768, -1] })).toThrow('Int16');
    expect(() => buildCategoricalSurfaceGeometry({ positions, indices, labels: [1, -2, -1] })).toThrow('Int16');
  });
});
