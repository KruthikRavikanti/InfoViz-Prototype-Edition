import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { SelectedHeatmapCell, VisualizationData } from './types/data';

const appState = vi.hoisted(() => ({
  data: null as VisualizationData | null,
  dataError: null as string | null,
  selectedCell: null as SelectedHeatmapCell | null,
  nullStatesSeen: 0,
}));

// SSR does not run the data-loading effect. Inject ready data so these checks
// exercise the real explorer, rather than passing on its loading screen.
vi.mock('react', async (importOriginal) => {
  const original = await importOriginal<typeof import('react')>();
  return {
    ...original,
    useState(initialState: unknown) {
      if (appState.dataError && typeof initialState === 'object' && initialState !== null
        && 'status' in initialState && initialState.status === 'loading') {
        return original.useState({ status: 'error', message: appState.dataError });
      }
      if (appState.data && typeof initialState === 'object' && initialState !== null
        && 'status' in initialState && initialState.status === 'loading') {
        return original.useState({ status: 'ready', data: appState.data });
      }
      // App's first null state is its selected heatmap cell. Keep subsequent
      // null states (focused upload, image modal, etc.) unchanged.
      if (initialState === null && appState.nullStatesSeen++ === 0 && appState.selectedCell) {
        return original.useState(appState.selectedCell);
      }
      return original.useState(initialState);
    },
  };
});

function visualizationFixture(): VisualizationData {
  return {
    rows: [
      { image_name: 'image_001.png', clip_rn50_ffa: 0.7, clip_rn50_eba: 0.4, clip_rn50_ppa: 0.2 },
      { image_name: 'image_002.png', clip_rn50_ffa: 0.3, clip_rn50_eba: 0.6, clip_rn50_ppa: 0.8 },
    ],
    scores: {
      ffa: { clip_rn50: 0.8 },
      eba: { clip_rn50: 0.6 },
      ppa: { clip_rn50: 0.7 },
      Overall: { clip_rn50: 0.7 },
    },
    modelRoiColumns: ['ffa', 'eba', 'ppa'].map((roi) => ({
      columnName: `clip_rn50_${roi}`, model: 'clip_rn50', roi,
    })),
    clustering: {
      voxelSummaries: [],
      voxelSummariesByMethod: { kmeans_euclidean: [], hierarchical_correlation: [] },
      visualSummary: null,
      visualView: null,
      visualSummaryByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
      visualViewByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
      dreamsimSummary: null,
      dreamsimView: null,
      dreamsimSummaryByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
      dreamsimViewByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
      imageCategories: new Map(),
      groundTruthPatients: [],
      groundTruthPatientsByMethod: { kmeans_euclidean: [], hierarchical_correlation: [] },
    },
  };
}

function renderExplorer(selectedCell: SelectedHeatmapCell | null = null, embedded = false): string {
  appState.data = visualizationFixture();
  appState.selectedCell = selectedCell;
  appState.nullStatesSeen = 0;
  vi.stubGlobal('window', { location: { hash: '', search: embedded ? '?embedded=1' : '' }, parent: embedded ? { postMessage: vi.fn() } : undefined });
  return renderToStaticMarkup(<App />);
}

afterEach(() => {
  appState.data = null;
  appState.dataError = null;
  appState.selectedCell = null;
  appState.nullStatesSeen = 0;
  vi.unstubAllGlobals();
});

describe('Cortex-Compass application', () => {
  it('uses the product name while loading the standalone workspace', () => {
    vi.stubGlobal('window', { location: { hash: '' } });
    const markup = renderToStaticMarkup(<App />);

    expect(markup).toContain('<h1>Loading Cortex-Compass</h1>');
    expect(markup).not.toContain('Loading the explorer');
  });

  it('opens the explorer directly with Cortex navigation and all model controls', () => {
    const markup = renderExplorer();

    expect(markup).toContain('<section class="visualization-panel" aria-label="Cortex-Compass workspace">');
    expect(markup).toContain('<h1>Cortex-Compass</h1>');
    expect(markup).toContain('Murty-185 (2 images)');
    expect(markup).toContain('Select a model and region to explore predictions.');
    expect(markup).toContain('aria-label="Cortex navigation"');
    expect(markup).toContain('Back to the Lab');
    expect(markup).toContain('aria-label="Model controls"');
    expect(markup).toContain('Search models');
    expect(markup).toContain('Rank models');
    expect(markup).toContain('Show scores');
    expect(markup).toContain('Compare selections');
    expect(markup).toContain('aria-label="Heatmap area"');
    expect(markup.match(/data-cell-id="(?:ffa|eba|ppa)__clip_rn50"/g)).toHaveLength(3);
    expect(markup).not.toContain('User Study');
    expect(markup).not.toContain('user-study');
    expect(markup).not.toContain('aria-label="Primary views"');
    expect(markup).not.toContain('<iframe');
    expect(markup).not.toContain('docs.google.com/forms');
  });

  it('preserves the explorer under the host header without a second navigation bar', () => {
    const markup = renderExplorer(null, true);

    expect(markup).toContain('<main class="app-shell app-shell--embedded">');
    expect(markup).toContain('<h1>Cortex-Compass</h1>');
    expect(markup).toContain('Murty-185 (2 images)');
    expect(markup).toContain('aria-label="Model controls"');
    expect(markup).toContain('aria-label="Heatmap area"');
    expect(markup).not.toContain('class="app-masthead"');
    expect(markup).not.toContain('Back to the Lab');
    expect(markup).not.toContain('<iframe');
  });

  it('uses the embedded shell while loading without removing standalone navigation based on a flag alone', () => {
    vi.stubGlobal('window', { location: { hash: '', search: '?embedded=1' }, parent: { postMessage: vi.fn() } });
    expect(renderToStaticMarkup(<App />)).toContain('<main class="app-shell app-shell--embedded">');
    vi.stubGlobal('window', { location: { hash: '', search: '?embedded=1' } });
    expect(renderToStaticMarkup(<App />)).toContain('<main class="app-shell">');
  });

  it('keeps error-screen return navigation outside the frame', () => {
    vi.stubGlobal('window', { location: { hash: '#session=bad', search: '?embedded=1' }, parent: { postMessage: vi.fn() } });
    const markup = renderToStaticMarkup(<App />);
    expect(markup).toContain('class="app-shell app-shell--embedded session-error-shell"');
    expect(markup).toContain('target="_top"');
    expect(markup).toContain('Return to Virtual Cortex');
  });

  it('offers the same top-level return when embedded public data cannot load', () => {
    appState.dataError = 'References unavailable';
    vi.stubGlobal('window', { location: { hash: '', search: '?embedded=1' }, parent: { postMessage: vi.fn() } });
    const markup = renderToStaticMarkup(<App />);
    expect(markup).toContain('<main class="app-shell app-shell--embedded">');
    expect(markup).toContain('References unavailable');
    expect(markup).toContain('target="_top"');
    expect(markup).toContain('Return to Virtual Cortex');
  });

  it('preserves the three analysis tabs and four cluster views when a model is selected', () => {
    const markup = renderExplorer({
      id: 'ffa__clip_rn50', model: 'clip_rn50', roi: 'ffa',
      score: 0.8, overallScore: 0.7, rankWithinRoi: 1,
    });

    expect(markup).toContain('<strong>Viewing:</strong> clip_rn50 · FFA · NSD-1000 predictions');
    expect(markup).toContain('class="detail-subtabs"');
    expect(markup).toContain('Image responses');
    expect(markup).toContain('Image clusters');
    expect(markup).toContain('Brain &amp; voxels');
    expect(markup).toContain('<h3>Predicted-response clusters</h3>');
    expect(markup).toContain('<h3>Visual-feature clusters</h3>');
    expect(markup).toContain('<h3>DreamSim similarity clusters</h3>');
    expect(markup).toContain('<h3>Measured fMRI clusters</h3>');
    expect(markup).not.toContain('<iframe');
  });
});
