export interface CategoricalSurfaceGeometryInput {
  positions: ArrayLike<number>;
  normals?: ArrayLike<number>;
  indices: ArrayLike<number>;
  /** One label per original mesh vertex; -1 means unsupported. */
  labels: ArrayLike<number>;
}

export interface CategoricalSurfaceGeometry {
  /** Non-indexed triangles, with the original mesh winding preserved. */
  positions: Float32Array;
  normals?: Float32Array;
  /** Constant on each output triangle, so categorical colors cannot blend. */
  clusterLabels: Int16Array;
  /** One owning original mesh vertex per output triangle (raycast faceIndex). */
  sourceVertexIndices: Uint32Array;
  triangleCount: number;
}

// Each original vertex owns the quad bounded by itself, its two incident edge
// midpoints and the face centroid. Split that quad without changing winding.
// These barycentric coordinates partition the original face into equal thirds.
const VERTEX_CELLS = [
  [
    [1, 0, 0], [0.5, 0.5, 0], [1 / 3, 1 / 3, 1 / 3],
    [1, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0, 0.5],
  ],
  [
    [0, 1, 0], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 1, 0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0],
  ],
  [
    [0, 0, 1], [0.5, 0, 0.5], [1 / 3, 1 / 3, 1 / 3],
    [0, 0, 1], [1 / 3, 1 / 3, 1 / 3], [0, 0.5, 0.5],
  ],
] as const;

/**
 * Display existing vertex labels as crisp categorical regions on the existing
 * cortical faces. A supported vertex owns only its barycentric third of each
 * incident face. Unsupported cells are omitted, exposing the gray cortex below.
 * This is a display partition, not a new measurement at the centroid, label
 * interpolation, spatial dilation, or a claim of support beyond these faces.
 */
export function buildCategoricalSurfaceGeometry({
  positions,
  normals,
  indices,
  labels,
}: CategoricalSurfaceGeometryInput): CategoricalSurfaceGeometry {
  if (positions.length % 3 !== 0) {
    throw new Error('Surface positions must contain complete xyz vertices.');
  }
  if (indices.length % 3 !== 0) {
    throw new Error('Surface indices must contain complete triangles.');
  }
  const vertexCount = positions.length / 3;
  if (labels.length !== vertexCount) {
    throw new Error('Surface labels must match the original mesh vertex count.');
  }
  if (normals && normals.length !== positions.length) {
    throw new Error('Surface normals must match the position array length.');
  }
  for (let i = 0; i < positions.length; i += 1) {
    if (!Number.isFinite(positions[i]) || (normals && !Number.isFinite(normals[i]))) {
      throw new Error('Surface positions and normals must be finite.');
    }
  }
  for (let i = 0; i < labels.length; i += 1) {
    if (!Number.isInteger(labels[i]) || labels[i] < -1 || labels[i] > 32767) {
      throw new Error('Surface labels must be -1 or a nonnegative Int16 cluster label.');
    }
  }

  let supportedCorners = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i];
    const b = indices[i + 1];
    const c = indices[i + 2];
    for (const index of [a, b, c]) {
      if (!Number.isInteger(index) || index < 0 || index >= vertexCount) {
        throw new Error(`Surface triangle ${i / 3} has an invalid vertex index.`);
      }
    }
    if (a === b || b === c || c === a) {
      throw new Error(`Surface triangle ${i / 3} has repeated vertex indices.`);
    }
    const abx = positions[b * 3] - positions[a * 3];
    const aby = positions[b * 3 + 1] - positions[a * 3 + 1];
    const abz = positions[b * 3 + 2] - positions[a * 3 + 2];
    const acx = positions[c * 3] - positions[a * 3];
    const acy = positions[c * 3 + 1] - positions[a * 3 + 1];
    const acz = positions[c * 3 + 2] - positions[a * 3 + 2];
    const crossX = aby * acz - abz * acy;
    const crossY = abz * acx - abx * acz;
    const crossZ = abx * acy - aby * acx;
    if (crossX === 0 && crossY === 0 && crossZ === 0) {
      throw new Error(`Surface triangle ${i / 3} has zero area.`);
    }
    supportedCorners += Number(labels[a] >= 0) + Number(labels[b] >= 0) + Number(labels[c] >= 0);
  }

  const triangleCount = supportedCorners * 2;
  const outputPositions = new Float32Array(triangleCount * 9);
  const outputNormals = normals ? new Float32Array(triangleCount * 9) : undefined;
  const clusterLabels = new Int16Array(triangleCount * 3);
  const sourceVertexIndices = new Uint32Array(triangleCount);
  let outputVertex = 0;
  let outputTriangle = 0;

  for (let i = 0; i < indices.length; i += 3) {
    const vertices = [indices[i], indices[i + 1], indices[i + 2]];
    for (let corner = 0; corner < 3; corner += 1) {
      const owner = vertices[corner];
      const label = labels[owner];
      if (label < 0) continue;
      sourceVertexIndices[outputTriangle++] = owner;
      sourceVertexIndices[outputTriangle++] = owner;
      for (const weights of VERTEX_CELLS[corner]) {
        for (let axis = 0; axis < 3; axis += 1) {
          const offset = outputVertex * 3 + axis;
          outputPositions[offset] = positions[vertices[0] * 3 + axis] * weights[0]
            + positions[vertices[1] * 3 + axis] * weights[1]
            + positions[vertices[2] * 3 + axis] * weights[2];
          if (outputNormals && normals) {
            // Keep the original affine normal field; the renderer normalizes
            // it after interpolation, as it does on the unsplit source face.
            outputNormals[offset] = normals[vertices[0] * 3 + axis] * weights[0]
              + normals[vertices[1] * 3 + axis] * weights[1]
              + normals[vertices[2] * 3 + axis] * weights[2];
          }
        }
        clusterLabels[outputVertex++] = label;
      }
    }
  }

  return {
    positions: outputPositions,
    normals: outputNormals,
    clusterLabels,
    sourceVertexIndices,
    triangleCount,
  };
}
