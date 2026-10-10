import { describe, expect, it, vi } from 'vitest';
import referenceIdentity from '../../public/data/reference_manifest.json';
import reviewedManifestText from '../../public/data/voxel_selectivity_atlas/manifest.json?raw';
import reviewedPointsText from '../../public/data/voxel_selectivity_atlas/clip_rn50/points.csv?raw';
import reviewedSummaryText from '../../public/data/voxel_selectivity_atlas/clip_rn50/summary.json?raw';
import {
  loadSelectivityAtlasAssets,
  loadSelectivityAtlasManifest,
  parseSelectivityAtlasPoints,
  SELECTIVITY_ATLAS_MANIFEST_PATH,
  validateSelectivityAtlasManifest,
  validateSelectivityAtlasSummary,
} from './selectivityAtlasAssets';

const modelNames = referenceIdentity.supported_models;
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
    referenceVersion: referenceIdentity.reference_version,
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
      meanSilhouette: 0.1 as number | null,
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

  it('rejects a substituted model even when model records remain internally consistent', () => {
    const manifest = rawManifest();
    const removed = manifest.supportedModels[0];
    manifest.supportedModels = ['unreviewed_model', ...manifest.supportedModels.slice(1)];
    delete manifest.models[removed];
    manifest.models.unreviewed_model = {
      pointsUrl: 'unreviewed_model/points.csv',
      summaryUrl: 'unreviewed_model/summary.json',
      pointsSha256: 'a'.repeat(64),
      summarySha256: 'b'.repeat(64),
      pointCount: 844,
    };
    expect(() => validateSelectivityAtlasManifest(manifest)).toThrow(/model coverage/);
  });

  it.each(['image_000.png', 'image_186.png', 'image_999.png'])('rejects non-reference image %s', (imageId) => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    for (const field of ['preferredImages', 'suppressedImages'] as const) {
      const summary = rawSummary();
      summary.clusters[0][field][0].imageId = imageId;
      expect(() => validateSelectivityAtlasSummary(summary, manifest, 'clip_rn50')).toThrow(/not canonical/);
    }
  });

  it('accepts the first and last canonical reference image IDs', () => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    const summary = rawSummary();
    summary.clusters[0].preferredImages[4].imageId = 'image_185.png';
    expect(validateSelectivityAtlasSummary(summary, manifest, 'clip_rn50').clusters[0].preferredImages)
      .toEqual(expect.arrayContaining([
        expect.objectContaining({ imageId: 'image_001.png' }),
        expect.objectContaining({ imageId: 'image_185.png' }),
      ]));
  });

  it.each([-1.01, 1.01, Number.NaN, Number.POSITIVE_INFINITY])('rejects invalid cluster mean silhouette %s', (score) => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    const summary = rawSummary();
    summary.clusters[0].meanSilhouette = score;
    expect(() => validateSelectivityAtlasSummary(summary, manifest, 'clip_rn50')).toThrow(/meanSilhouette/);
  });

  it.each([-1, 1, null])('accepts cluster mean silhouette %s', (score) => {
    const manifest = validateSelectivityAtlasManifest(rawManifest());
    const summary = rawSummary();
    summary.clusters[0].meanSilhouette = score;
    expect(validateSelectivityAtlasSummary(summary, manifest, 'clip_rn50').clusters[0].meanSilhouette).toBe(score);
  });

  it('loads the unchanged reviewed manifest only after its pinned digest succeeds', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(reviewedManifestText));
    const controller = new AbortController();
    const manifest = await loadSelectivityAtlasManifest(fetcher, controller.signal);
    expect(manifest.supportedModels).toEqual(modelNames);
    expect(fetcher).toHaveBeenCalledExactlyOnceWith(SELECTIVITY_ATLAS_MANIFEST_PATH, { signal: controller.signal });
  });

  it('rejects manifest tampering before requesting any child asset', async () => {
    // Even whitespace-only changes retain the JSON identity but invalidate the reviewed bytes.
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(`${reviewedManifestText} `));
    await expect(loadSelectivityAtlasAssets('clip_rn50', fetcher)).rejects.toThrow(/manifest failed its SHA-256 check/);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('loads the existing reviewed CLIP RN50 manifest, summary, and voxel points together', async () => {
    const fetcher = vi.fn<typeof fetch>().mockImplementation(async (url) => new Response(
      String(url).endsWith('/manifest.json') ? reviewedManifestText
        : String(url).endsWith('/summary.json') ? reviewedSummaryText : reviewedPointsText,
    ));
    const assets = await loadSelectivityAtlasAssets('clip_rn50', fetcher);
    expect(assets.summary.model).toBe('clip_rn50');
    expect(assets.points).toHaveLength(844);
    expect(assets.summary.clusters).toHaveLength(14);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it.each(['points', 'summary'])('continues rejecting tampered %s after the manifest is verified', async (kind) => {
    const fetcher = vi.fn<typeof fetch>().mockImplementation(async (url) => {
      const isSummary = String(url).endsWith('/summary.json');
      const text = String(url).endsWith('/manifest.json') ? reviewedManifestText
        : isSummary ? reviewedSummaryText : reviewedPointsText;
      const tampered = kind === 'summary' ? isSummary : String(url).endsWith('/points.csv');
      return new Response(tampered ? `${text} ` : text);
    });
    await expect(loadSelectivityAtlasAssets('clip_rn50', fetcher)).rejects.toThrow(/failed its SHA-256 check/);
  });

  it('allows a later retry after a manifest integrity failure', async () => {
    const fetcher = vi.fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(`${reviewedManifestText} `))
      .mockResolvedValueOnce(new Response(reviewedManifestText));
    await expect(loadSelectivityAtlasManifest(fetcher)).rejects.toThrow(/SHA-256/);
    await expect(loadSelectivityAtlasManifest(fetcher)).resolves.toHaveProperty('subject', 's1');
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('does not request a manifest for an already aborted caller', async () => {
    const fetcher = vi.fn<typeof fetch>();
    const controller = new AbortController();
    controller.abort();
    await expect(loadSelectivityAtlasManifest(fetcher, controller.signal)).rejects.toHaveProperty('name', 'AbortError');
    expect(fetcher).not.toHaveBeenCalled();
  });
});
