import type { CanonicalRoi } from './session';

export type BrainSurfaceKind = 'inflated' | 'pial';
export type BrainHemisphere = 'lh' | 'rh';

export type BrainSurfaceHemisphere = {
  node: string;
  vertexCount: number;
  faceCount: number;
};

export type BrainSurfaceAsset = {
  url: string;
  sha256: string;
  normalsIncluded: boolean;
  hemispheres: Record<BrainHemisphere, BrainSurfaceHemisphere>;
};

export type BrainMapManifest = {
  schemaVersion: 1;
  subjectId: 's1';
  responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  fLocThreshold: 7;
  fLocThresholdOperator: '>';
  responseOrdering: 'zyx-c-order';
  mapper: 'linenn';
  normalizationPolicy: 'per-surface-vertex';
  volumeShapeXyz: [83, 104, 81];
  subject: {
    id: 'nsd-subj01';
    label: string;
    responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  };
  volumeSpace: {
    name: string;
    shapeXyz: [number, number, number];
    sourceAxisOrder: string;
    responseOrdering: string;
  };
  meshSpace: {
    source: string;
    gltfTransform: string;
    mapper: string;
    vertexOrderingPreserved: boolean;
    recentered: boolean;
  };
  surfaces: Partial<Record<BrainSurfaceKind, BrainSurfaceAsset>>;
  projectionUrl: string;
  projectionSha256: string;
  rois: Record<CanonicalRoi, {
    label: string;
    responseCount: number;
    mappedResponseCount: number;
    unmappedResponseIndices: number[];
    orderedCoordinateSha256: string;
    color: string;
  }>;
};

export type BrainProjectionMapping = {
  responseIndices: number[];
  vertexIndices: number[];
  weights: number[];
};

export type BrainProjectionRoi = {
  responseCount: number;
  mappedResponseCount: number;
  volumeCoordinatesZyx: Array<[number, number, number]>;
  volumeLinearIndicesXyz: number[];
  unmappedResponseIndices: number[];
  hemispheres: Array<BrainHemisphere | 'bilateral' | 'unmapped'>;
  mappings: Record<BrainHemisphere, BrainProjectionMapping>;
};

export type BrainProjection = {
  schemaVersion: 1;
  subjectId: 's1';
  responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  fLocThreshold: 7;
  fLocThresholdOperator: '>';
  responseOrdering: 'zyx-c-order';
  mapper: 'linenn';
  normalizationPolicy: 'per-surface-vertex';
  volumeShapeXyz: [83, 104, 81];
  rois: Record<CanonicalRoi, BrainProjectionRoi>;
};

export type BrainMapAssets = {
  manifest: BrainMapManifest;
  projection: BrainProjection;
};

/** Raw NSD Subject 1 predictions for the immutable Murty-185 reference set. */
export type BrainMapReferenceSet = {
  model: string;
  referenceVersion: string;
  responseSpace: 'nsd_1000_subject1_floc_t7_v1';
  stimuli: Array<{
    imageName: string;
    imageUrl: string;
    roiResponses: Record<CanonicalRoi, number[]>;
  }>;
};

export type BrainMapHoverDatum = {
  hemisphere: BrainHemisphere;
  vertexIndex: number;
  surfaceResponse: number;
  overlappingRois: CanonicalRoi[];
  dominantSource: {
    roi: CanonicalRoi;
    responseIndex: number;
    response: number;
    mappingWeight: number;
  };
  volumeCoordinateZyx: [number, number, number];
  volumeLinearIndexXyz: number;
  canvasX: number;
  canvasY: number;
};

export type BrainSurfaceVertexDatum = {
  surfaceResponse: number;
  overlappingRois: CanonicalRoi[];
  dominantSource: BrainMapHoverDatum['dominantSource'];
};

export type BrainHemisphereProjection = {
  values: Float32Array;
  mapped: Uint8Array;
  vertexData: Array<BrainSurfaceVertexDatum | null>;
};
