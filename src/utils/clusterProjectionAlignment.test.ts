import { csvParse } from 'd3';
import { describe, expect, it } from 'vitest';
import type { ClusterPoint, ClusterSummary, ClusterView, ProjectionAxisAlignment } from '../types/data';
import { buildClusterView } from './clustering';
import { inferClusterProjectionAlignment } from './clusterProjectionAlignment';
import kmeansVisualCsv from '../../public/data/visual_clusters/visual_clusters.csv?raw';
import hierarchicalVisualCsv from '../../public/data/visual_clusters_hierarchical_corr/visual_clusters.csv?raw';
import referenceIdentity from '../../public/data/reference_manifest.json';

const kmeansVoxelCsvs = import.meta.glob<string>('../../public/data/voxel_clusters/*_clusters.csv', {
  query: '?raw', import: 'default', eager: true,
});
const hierarchicalVoxelCsvs = import.meta.glob<string>('../../public/data/voxel_clusters_hierarchical_corr/*_clusters.csv', {
  query: '?raw', import: 'default', eager: true,
});

function summary(count: number): ClusterSummary {
  return { nImages: count, bestK: 2, silhouette: null, clusterSizes: {} };
}

function point(imageName: string, x: number, y: number, clusterLabel = '0'): ClusterPoint {
  return {
    imageName, imageUrl: `/images/${imageName}`, clusterLabel, value: 0, rank: 1,
    pcaX: x, pcaY: y, plotX: x, plotY: y, tsneX: null, tsneY: null, distanceToCentroid: null,
  };
}

function view(points = [point('a.png', 2, -3), point('b.png', -4, 7), point('c.png', 9, 5)]): ClusterView {
  return buildClusterView(summary(points.length), points)!;
}

function transformed(source: ClusterView, signs: ProjectionAxisAlignment): ClusterView {
  return view(source.points.map((p) => point(p.imageName, p.pcaX! * signs.xSign, p.pcaY! * signs.ySign, 'different')));
}

function csvView(text: string): ClusterView {
  const number = (value: string | undefined): number | null => value === undefined || value === '' ? null : Number(value);
  const points = csvParse(text).map((row, index): ClusterPoint => ({
    imageName: row.image_name!, imageUrl: `/images/${row.image_name}`, clusterLabel: row.cluster_label!,
    value: number(row.distance_to_centroid) ?? 0, rank: index + 1,
    pcaX: number(row.pca_x), pcaY: number(row.pca_y), plotX: number(row.plot_x), plotY: number(row.plot_y),
    tsneX: number(row.tsne_x), tsneY: number(row.tsne_y), distanceToCentroid: number(row.distance_to_centroid),
  }));
  return buildClusterView(summary(185), points)!;
}

describe('hierarchical upload PCA axis alignment', () => {
  it.each([
    { xSign: 1, ySign: 1 }, { xSign: 1, ySign: -1 },
    { xSign: -1, ySign: 1 }, { xSign: -1, ySign: -1 },
  ] as ProjectionAxisAlignment[])('validates independent sign flips %o using image IDs, not row order', (signs) => {
    const source = view();
    const target = transformed(source, signs);
    target.points.reverse();
    expect(inferClusterProjectionAlignment(source, target)).toEqual(signs);
  });

  it('does not alter reference coordinates, cluster membership, groups, or representative distances', () => {
    const source = view();
    const target = transformed(source, { xSign: 1, ySign: -1 });
    const before = JSON.stringify({ source, target });
    inferClusterProjectionAlignment(source, target);
    expect(JSON.stringify({ source, target })).toBe(before);
    expect(target.groups[0].label).toBe('different');
  });

  it('accepts serialization/recomputation noise but not an unverified arbitrary transform', () => {
    const source = view();
    const target = transformed(source, { xSign: -1, ySign: 1 });
    target.points[0].pcaX! += 2.6e-7;
    target.points[0].plotX = target.points[0].pcaX;
    expect(inferClusterProjectionAlignment(source, target)).toEqual({ xSign: -1, ySign: 1 });
    target.points[0].pcaX! += .001;
    target.points[0].plotX = target.points[0].pcaX;
    expect(inferClusterProjectionAlignment(source, target)).toBeNull();
  });

  it.each(['shift', 'rotation', 'scale', 'swapped-axes'])('rejects %s rather than fitting the uploads to changed geometry', (kind) => {
    const source = view();
    const target = view(source.points.map((p) => {
      const x = p.pcaX!, y = p.pcaY!;
      if (kind === 'shift') return point(p.imageName, x + 1, y - 2);
      if (kind === 'rotation') return point(p.imageName, .8 * x - .6 * y, .6 * x + .8 * y);
      if (kind === 'scale') return point(p.imageName, x * 1.2, y);
      return point(p.imageName, y, x);
    }));
    expect(inferClusterProjectionAlignment(source, target)).toBeNull();
  });

  it('rejects missing, mismatched, duplicate, or incomplete reference image sets', () => {
    const source = view();
    const missing = view(source.points.slice(1));
    expect(inferClusterProjectionAlignment(source, missing)).toBeNull();
    const mismatch = view();
    mismatch.points[1].imageName = 'unrelated.png';
    expect(inferClusterProjectionAlignment(source, mismatch)).toBeNull();
    const duplicate = view();
    duplicate.points[1].imageName = duplicate.points[0].imageName;
    expect(inferClusterProjectionAlignment(source, duplicate)).toBeNull();
    const incomplete = view();
    incomplete.summary!.nImages = 185;
    expect(inferClusterProjectionAlignment(source, incomplete)).toBeNull();
    const unknownReferenceCount = view();
    unknownReferenceCount.summary = null;
    expect(inferClusterProjectionAlignment(source, unknownReferenceCount)).toBeNull();
    expect(inferClusterProjectionAlignment(null, source)).toBeNull();
  });

  it.each([null, Number.NaN, Number.POSITIVE_INFINITY])('rejects invalid PCA coordinate %s', (invalid) => {
    const source = view();
    const target = view();
    target.points[1].pcaX = invalid;
    expect(inferClusterProjectionAlignment(source, target)).toBeNull();
  });

  it('rejects t-SNE and plot geometry that is not the displayed PCA coordinates', () => {
    const source = view();
    const target = view();
    target.coordinateMode = 'tsne';
    expect(inferClusterProjectionAlignment(source, target)).toBeNull();
    target.coordinateMode = 'plot';
    target.points[0].plotY! += 3;
    expect(inferClusterProjectionAlignment(source, target)).toBeNull();
    target.points[0].plotY = target.points[0].pcaY;
    target.coordinateMode = 'pca';
    expect(inferClusterProjectionAlignment(source, target)).toEqual({ xSign: 1, ySign: 1 });
  });

  it('rejects an axis with no orientational evidence', () => {
    const source = view([point('a.png', 0, 1), point('b.png', 0, 2), point('c.png', 0, 3)]);
    expect(inferClusterProjectionAlignment(source, source)).toBeNull();
  });

  it('validates every complete published model/ROI reference geometry', () => {
    const expectedCount = referenceIdentity.supported_models.length * referenceIdentity.rois.length;
    expect(Object.keys(kmeansVoxelCsvs)).toHaveLength(expectedCount);
    expect(Object.keys(hierarchicalVoxelCsvs)).toHaveLength(expectedCount);
    for (const model of referenceIdentity.supported_models) {
      for (const roi of referenceIdentity.rois) {
        // Full basename boundaries distinguish AlexNet/CORnet-S from VOne
        // variants; suffix matching would incorrectly count both families.
        const identity = `murtylab_${model}_nsd_1000_${roi}_`;
        const matches = (path: string) => path.split('/').pop()?.startsWith(identity);
        expect(Object.keys(kmeansVoxelCsvs).filter(matches)).toHaveLength(1);
        expect(Object.keys(hierarchicalVoxelCsvs).filter(matches)).toHaveLength(1);
      }
    }
    for (const [hierarchicalPath, text] of Object.entries(hierarchicalVoxelCsvs)) {
      const sourcePath = hierarchicalPath.replace('/voxel_clusters_hierarchical_corr/', '/voxel_clusters/');
      const source = csvView(kmeansVoxelCsvs[sourcePath]);
      const target = csvView(text);
      expect(inferClusterProjectionAlignment(source, target), hierarchicalPath).not.toBeNull();
      if (hierarchicalPath.includes('clip_rn50_nsd_1000_eba_')) {
        expect(inferClusterProjectionAlignment(source, target)).toEqual({ xSign: 1, ySign: -1 });
      }
    }
  });

  it('validates the actual visual-feature display axes despite tiny recomputation differences', () => {
    const source = csvView(kmeansVisualCsv);
    const target = csvView(hierarchicalVisualCsv);
    expect(source.coordinateMode).toBe('plot');
    expect(target.coordinateMode).toBe('plot');
    expect(inferClusterProjectionAlignment(source, target)).toEqual({ xSign: -1, ySign: 1 });
  });
});
