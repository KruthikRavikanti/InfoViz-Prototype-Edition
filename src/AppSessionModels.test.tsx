import { renderToStaticMarkup } from 'react-dom/server';
import type { ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { VisualizationView } from './components/VisualizationView';
import type { CortexComparisonSession } from './types/session';
import type { SelectedHeatmapCell, VisualizationData } from './types/data';
import { rawSessionFixture } from './test/sessionFixture';
import { validateComparisonSession } from './services/cortexSession';

const harness = vi.hoisted(() => ({
  data: null as VisualizationData | null,
  session: null as CortexComparisonSession | null,
  cell: null as SelectedHeatmapCell | null,
  view: null as ComponentProps<typeof VisualizationView> | null,
  changes: {} as Record<string, unknown>,
  nullStates: 0,
  falseStates: 0,
}));

vi.mock('./components/Header', () => ({ Header: () => null }));
vi.mock('./components/VisualizationView', () => ({
  VisualizationView: (props: ComponentProps<typeof VisualizationView>) => {
    harness.view = props;
    return null;
  },
}));

vi.mock('react', async (importOriginal) => {
  const original = await importOriginal<typeof import('react')>();
  return {
    ...original,
    useState(initial: unknown) {
      let value = initial;
      let key: string | undefined;
      if (initial && typeof initial === 'object' && 'status' in initial) value = { status: 'ready', data: harness.data };
      if (typeof initial === 'function') value = { status: 'ready', session: harness.session };
      if (initial === null && harness.nullStates++ === 0) { value = harness.cell; key = 'cell'; }
      if (initial === 'ffa') key = 'roi';
      if (initial === '') { value = 'not-a-prepared-model'; key = 'search'; }
      if (initial === false && harness.falseStates++ === 1) key = 'compareMode';
      if (Array.isArray(initial)) key = 'compareCells';
      return [value, (next: unknown) => { if (key) harness.changes[key] = next; }];
    },
  };
});

const models = ['resnet50', 'efficient_net', 'clip_rn50', 'vgg16_imagenet1kv1'];

function renderSession(cell: SelectedHeatmapCell | null) {
  const raw = rawSessionFixture();
  raw.comparison.model = 'resnet50';
  raw.source_request = { model: 'resnet50', dataset: 'nsd_1000', roi: 'ffa' };
  harness.session = validateComparisonSession(raw, 'http://core.example');
  harness.session.modelResults = Object.fromEntries(models.slice(1).map((model) => [model, {
    roiResults: structuredClone(harness.session!.roiResults),
  }]));
  const modelScores = Object.fromEntries(models.map((model, index) => [model, .9 - index / 10]));
  const scores = { ffa: modelScores, eba: modelScores, ppa: modelScores, Overall: modelScores };
  harness.data = { rows: [], scores, normalizedScores: scores, modelRoiColumns: [], clustering: {
    voxelSummaries: [], voxelSummariesByMethod: { kmeans_euclidean: [], hierarchical_correlation: [] },
    visualSummary: null, visualView: null,
    visualSummaryByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
    visualViewByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
    dreamsimSummary: null, dreamsimView: null,
    dreamsimSummaryByMethod: { kmeans_euclidean: null, hierarchical_correlation: null },
    dreamsimViewByMethod: { kmeans_euclidean: null, hierarchical_correlation: null }, imageCategories: new Map(),
    groundTruthPatients: [], groundTruthPatientsByMethod: { kmeans_euclidean: [], hierarchical_correlation: [] },
  } };
  harness.cell = cell;
  vi.stubGlobal('window', { location: { hash: '', search: '' } });
  renderToStaticMarkup(<App />);
  return harness.view!;
}

afterEach(() => {
  harness.data = null;
  harness.session = null;
  harness.cell = null;
  harness.view = null;
  harness.changes = {};
  harness.nullStates = 0;
  harness.falseStates = 0;
  vi.unstubAllGlobals();
});

describe('prepared upload model selection', () => {
  it('marks all four prepared models without changing the original Lab model', () => {
    const view = renderSession(null);
    expect(view.sessionCellIds.size).toBe(12);
    for (const model of models) for (const roi of ['ffa', 'eba', 'ppa']) {
      expect(view.sessionCellIds.has(`${roi}__${model}`)).toBe(true);
    }
    expect(view.session!.comparison.model).toBe('resnet50');
    expect(view.session!.sourceRequest.model).toBe('resnet50');
  });

  it('keeps the selected ROI when switching prepared models, even when search hides the model', () => {
    const view = renderSession({ id: 'eba__clip_rn50', model: 'clip_rn50', roi: 'eba',
      score: .7, normalizedScore: .7, overallScore: .7, rankWithinRoi: 3 });
    expect(view.visibleModels).toEqual([]);
    view.onSelectSessionModel('efficient_net');
    expect(harness.changes.cell).toMatchObject({ model: 'efficient_net', roi: 'eba', id: 'eba__efficient_net', rankWithinRoi: 2 });
    expect(harness.changes.roi).toBe('eba');
    expect(harness.changes.search).toBe('');
    expect(harness.changes.compareMode).toBe(false);
    expect(harness.changes.compareCells).toEqual([]);
    expect(view.session!.sourceRequest.model).toBe('resnet50');
  });

  it('uses the original Lab region when no cell is selected and refuses unprepared model switches', () => {
    const view = renderSession(null);
    view.onSelectSessionModel('efficient_net');
    expect(harness.changes.cell).toMatchObject({ model: 'efficient_net', roi: 'ffa' });
    harness.changes = {};
    view.onSelectSessionModel('alexnet');
    expect(harness.changes).toEqual({});
  });
});
