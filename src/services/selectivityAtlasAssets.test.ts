import { describe, expect, it } from 'vitest';
import {
  parseSelectivityAtlasPoints,
  validateSelectivityAtlasManifest,
  validateSelectivityAtlasSummary,
} from './selectivityAtlasAssets';

const modelNames = ['clip_rn50', ...Array.from({ length: 25 }, (_, index) => `model_${index}`)];
const caveat = 'Fixed k=14 is inherited from the mentor workflow and is re-audited here only as exploratory.';
const method = {
  id: 'mentor-selectivity-k14-v1',
  objectClustered: 'voxels',
  normalization: 'per-voxel z-score across images (ddof=0, epsilon=1e-9)',
  pcaComponents: 35,
  l2Normalization: true,
  clustering: 'K-means in L2-normalized PCA selectivity space',
  clusterCount: 14,
  kmeansNInit: 25,
  kmeansRandomState: 0,
  embedding: '3D UMAP (n_neighbors=25, min_dist=0.1, metric=cosine, random_state=42); display only',
  embeddingUsedForClustering: false,
};

function rawManifest() {
  return {
    schemaVersion: 1,
    artifact: 'voxel-selectivity-atlas',
    dataset: 'nsd_1000',
    stimulusSet: 'murty185',
    subject: 's1',
    responseSpace: 'nsd_1000_subject1_floc_t7_v1',
    responseKind: 'model_predictions',
    embeddingIsAnatomical: false,
    imageCount: 185,
    voxelCount: 844,
    clusterCount: 14,
    roiOrder: ['ffa', 'eba', 'ppa'],
    rois: {
      ffa: { offset: 0, count: 156 },
      eba: { offset: 156, count: 518 },
      ppa: { offset: 674, count: 170 },
    },
    method,
    k14StabilityCaveat: caveat,
    supportedModels: modelNames,
    referenceVersion: 'ed023e59bd270e210bb8350d348eaaf223d3992214adeb2163a496cc808cefa7',
    imageIdsSha256: '1057a45119e774b1bd9c3462664d2d8003ca8d6a1b42c1691f084ce10533a7ca',
    models: Object.fromEntries(modelNames.map((model) => [model, {
      pointsUrl: `${model}/points.csv`,
      summaryUrl: `${model}/summary.json`,
      pointsSha256: 'a'.repeat(64),
      summarySha256: 'b'.repeat(64),
      pointCount: 844,
    }])),
  };
}

function roiForIndex(index: number): { roi: 'ffa' | 'eba' | 'ppa'; roiIndex: number } {
  if (index < 156) return { roi: 'ffa', roiIndex: index };
  if (index < 674) return { roi: 'eba', roiIndex: index - 156 };
  return { roi: 'ppa', roiIndex: index - 674 };
}

function rawSummary() {
  const clusters = Array.from({ length: 14 }, (_, label) => {
    const indices = Array.from({ length: 844 }, (_, index) => index).filter((index) => index % 14 === label);
    const roiCounts = { ffa: 0, eba: 0, ppa: 0 };
    indices.forEach((index) => { roiCounts[roiForIndex(index).roi] += 1; });
    const images = Array.from({ length: 5 }, (_, index) => ({
      imageId: `image_${String(index + 1).padStart(3, '0')}.png`,
      meanZ: 1 - index / 5,
    }));
    return {
      label,
      size: indices.length,
      roiCounts,
      meanSilhouette: 0.1,
      preferredImages: images,
      suppressedImages: images.map((image) => ({ ...image, meanZ: -image.meanZ })),
    };
  });
  const scores = [0.61, 0.72, 0.68, 0.75, 0.7, 0.66];
  return {
    schemaVersion: 1,
    model: 'clip_rn50',
    dataset: 'nsd_1000',
    stimulusSet: 'murty185',
    subject: 's1',
    responseSpace: 'nsd_1000_subject1_floc_t7_v1',
    responseKind: 'model_predictions',
    embeddingIsAnatomical: false,
    imageCount: 185,
    voxelCount: 844,
    clusterCount: 14,
    silhouette: 0.15,
    pcaExplainedVarianceRatioSum: 0.9,
    splitHalfStability: {
      iterations: 6,
      randomSeed: 42,
      adjustedRandIndices: scores,
      meanAdjustedRandIndex: scores.reduce((sum, score) => sum + score, 0) / scores.length,
      minAdjustedRandIndex: Math.min(...scores),
      threshold: 0.7,
      meanPassesThreshold: false,
      allPassThreshold: false,
    },
    method,
    k14StabilityCaveat: caveat,
    sourceFiles: ['ffa', 'eba', 'ppa'].map((roi) => ({
      roi,
      fileName: `murtylab_clip_rn50_nsd_1000_${roi}_20260408164117065Z.csv`,
      sha256: 'e'.repeat(64),
    })),
    clusters,
  };
}

function pointsCsv(): string {
  const header = 'voxel_index,roi,roi_voxel_index,cluster_label,umap_x,umap_y,umap_z,distance_to_centroid';
  const rows = Array.from({ length: 844 }, (_, index) => {
    const { roi, roiIndex } = roiForIndex(index);
    return `${index},${roi},${roiIndex},${index % 14},${index / 10},${-index / 20},${index / 30},0.25`;
  });
  return `${header}\n${rows.join('\n')}\n`;
}

describe('voxel-selectivity atlas asset contract', () => {
  it('validates the reviewed Subject 1 method and all 844 ordered voxel points', () => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    const summary = validateSelectivityAtlasSummary(rawSummary(), manifest, 'clip_rn50');
    const points = parseSelectivityAtlasPoints(pointsCsv(), manifest, summary);

    expect(points).toHaveLength(844);
    expect(points[0]).toMatchObject({ roi: 'ffa', roiVoxelIndex: 0, clusterLabel: 0 });
    expect(points[156]).toMatchObject({ roi: 'eba', roiVoxelIndex: 0 });
    expect(points[674]).toMatchObject({ roi: 'ppa', roiVoxelIndex: 0 });
    expect(summary.splitHalfStability.meanPassesThreshold).toBe(false);
  });

  it('rejects a summary whose reported split-half result is internally inconsistent', () => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    const summary = rawSummary();
    summary.splitHalfStability.meanPassesThreshold = true;
    expect(() => validateSelectivityAtlasSummary(summary, manifest, 'clip_rn50'))
      .toThrow(/split-half stability summary is inconsistent/);
  });

  it('rejects a point that breaks the audited ROI response ordering', () => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    const summary = validateSelectivityAtlasSummary(rawSummary(), manifest, 'clip_rn50');
    const invalid = pointsCsv().replace('\n156,eba,0,', '\n156,ffa,0,');
    expect(() => parseSelectivityAtlasPoints(invalid, manifest, summary)).toThrow(/out of range/);
  });
});
