import reviewedManifestText from '../../public/data/voxel_selectivity_atlas/manifest.json?raw';
import reviewedPointsText from '../../public/data/voxel_selectivity_atlas/clip_rn50/points.csv?raw';
import reviewedSummaryText from '../../public/data/voxel_selectivity_atlas/clip_rn50/summary.json?raw';
import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  parseSelectivityAtlasPoints,
  validateSelectivityAtlasManifest,
  validateSelectivityAtlasSummary,
} from '../services/selectivityAtlasAssets';
import type { SelectivityAtlasAssets } from '../types/selectivityAtlas';
import { VoxelSelectivityAtlasDialog } from './VoxelSelectivityAtlasDialog';

const dialogState = vi.hoisted(() => ({ open: false, assets: null as SelectivityAtlasAssets | null }));

vi.mock('../generated/referenceCoverage', async (importOriginal) => {
  const original = await importOriginal<typeof import('../generated/referenceCoverage')>();
  return { ...original, REFERENCE_COVERAGE: { ...original.REFERENCE_COVERAGE, stochasticModels: ['vone_alexnet'] } };
});

vi.mock('react', async (importOriginal) => {
  const original = await importOriginal<typeof import('react')>();
  return {
    ...original,
    useState(initialState: unknown) {
      if (dialogState.open && initialState === false) return original.useState(true);
      if (dialogState.assets && typeof initialState === 'object' && initialState !== null
        && 'status' in initialState && initialState.status === 'loading') {
        return original.useState({ status: 'ready', assets: dialogState.assets });
      }
      return original.useState(initialState);
    },
  };
});

vi.mock('react-dom', async (importOriginal) => ({
  ...await importOriginal<typeof import('react-dom')>(),
  createPortal: (children: ReactNode) => children,
}));

afterEach(() => {
  dialogState.open = false;
  dialogState.assets = null;
  vi.unstubAllGlobals();
});

function referenceAssets(): SelectivityAtlasAssets {
  const manifest = validateSelectivityAtlasManifest(JSON.parse(reviewedManifestText));
  const summary = validateSelectivityAtlasSummary(JSON.parse(reviewedSummaryText), manifest, 'clip_rn50');
  const points = parseSelectivityAtlasPoints(reviewedPointsText, manifest, summary);
  return { manifest, summary, points };
}

describe('voxel selectivity dialog', () => {
  it('discloses stochastic reference sampling inside the selectivity dialog', () => {
    dialogState.open = true;
    vi.stubGlobal('document', { body: {} });
    const markup = renderToStaticMarkup(<VoxelSelectivityAtlasDialog model="vone_alexnet" available />);
    expect(markup).toContain('reference predictions are a seeded sample; repeated predictions may vary');
  });
  it('keeps the heavy viewer behind its accessible, availability-guarded launcher', () => {
    const markup = renderToStaticMarkup(<VoxelSelectivityAtlasDialog model="clip_rn50" available={false} />);
    expect(markup).toContain('aria-haspopup="dialog"');
    expect(markup).toContain('aria-expanded="false"');
    expect(markup).toContain('disabled=""');
    expect(markup).not.toContain('role="dialog"');
  });

  it('preserves the reference-only science and explains similarity axes and keyboard inspection', () => {
    dialogState.open = true;
    dialogState.assets = referenceAssets();
    vi.stubGlobal('document', { body: {} });
    const markup = renderToStaticMarkup(<VoxelSelectivityAtlasDialog model="clip_rn50" available />);
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('185 Murty reference images · uploads not included');
    expect(markup).toContain('Visibility only; clusters use all three regions.');
    expect(markup).toContain('14 exploratory clusters');
    expect(markup).toContain('UMAP provides the 3D display only.');
    expect(markup).toContain('Its axes have arbitrary orientation, not anatomical directions or activity units.');
    expect(markup).toContain('Keyboard controls');
    expect(markup).toContain('id="selectivity-atlas-keyboard-instructions"');
    expect(markup).toContain('Home / End jump to the first / last. Enter or Space selects the voxel’s cluster.');
    expect(markup).toContain('Reset view');
    expect(markup.match(/aria-label="Cluster C\d+, \d+ voxels"/g)).toHaveLength(14);
    expect(markup).toContain('<details class="selectivity-atlas-method">');
    expect(markup).toContain('<details class="selectivity-atlas-keyboard-help">');
  });
});
