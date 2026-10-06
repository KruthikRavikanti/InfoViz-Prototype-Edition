import { Children, isValidElement, type ComponentProps, type ReactElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import type { ClusteringData, ModelRoiColumn, SelectedHeatmapCell, WideCsvRow } from '../types/data';
import { CompareDrawer } from './CompareDrawer';
import { ControlsBar } from './ControlsBar';
import { DetailsPanel } from './DetailsPanel';
import { HelpBox } from './HelpBox';

const cells: SelectedHeatmapCell[] = [
  { id: 'ffa__clip_rn50', roi: 'ffa', model: 'clip_rn50', score: 0.8, rankWithinRoi: 1, overallScore: 0.75 },
  { id: 'ffa__resnet50', roi: 'ffa', model: 'resnet50', score: 0.6, rankWithinRoi: 2, overallScore: 0.55 },
];

// The rankings have Spearman rho 0: [1, 2, 3, 4] versus [2, 4, 1, 3].
const rows: WideCsvRow[] = [
  { image_name: 'reference-a.png', clip_rn50_ffa: 4, resnet50_ffa: 3 },
  { image_name: 'reference-b.png', clip_rn50_ffa: 3, resnet50_ffa: 1 },
  { image_name: 'reference-c.png', clip_rn50_ffa: 2, resnet50_ffa: 4 },
  { image_name: 'reference-d.png', clip_rn50_ffa: 1, resnet50_ffa: 2 },
];

const columns: ModelRoiColumn[] = [
  { columnName: 'clip_rn50_ffa', model: 'clip_rn50', roi: 'ffa' },
  { columnName: 'resnet50_ffa', model: 'resnet50', roi: 'ffa' },
];

const clustering: ClusteringData = {
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
};

function controlsProps(): ComponentProps<typeof ControlsBar> {
  return {
    rois: ['ffa', 'eba', 'ppa'], rankingSystem: 'roi', selectedRankingRoi: 'eba',
    modelSearch: 'clip', showScoreLabels: true, compareMode: false, compareSelectionCount: 1,
    onRankingSystemChange: vi.fn(), onSelectedRankingRoiChange: vi.fn(), onModelSearchChange: vi.fn(),
    onShowScoreLabelsChange: vi.fn(), onCompareModeChange: vi.fn(), onResetControls: vi.fn(),
  };
}

type ControlProps = {
  children?: ReactNode;
  type?: string;
  value?: string;
  checked?: boolean;
  onChange?: (event: { target: { value: string; checked: boolean } }) => void;
  onClick?: () => void;
};

// ControlsBar has no hooks; inspecting its elements lets us exercise the
// actual callbacks without introducing a DOM dependency into the SSR suite.
function nativeElements(tree: ReactNode): ReactElement<ControlProps>[] {
  return Children.toArray(tree).flatMap((child) => {
    if (!isValidElement<ControlProps>(child)) return [];
    return [child, ...nativeElements(child.props.children)];
  });
}

function closedDetails(markup: string, className: string): string {
  const match = markup.match(new RegExp(`<details class="${className}">([\\s\\S]*?)</details>`));
  expect(match, `${className} remains available and collapsed`).not.toBeNull();
  return match![1];
}

const detailsProps: ComponentProps<typeof DetailsPanel> = {
  heatmapCells: cells, imageCount: rows.length, clustering, modelRoiColumns: columns,
  onSelectCell: vi.fn(), rows, selectedCell: null, session: null,
  focusedStimulusId: null, onFocusStimulus: vi.fn(),
};

const compareProps: ComponentProps<typeof CompareDrawer> = {
  compareMode: true, compareCells: cells, clustering, heatmapCells: cells,
  rows, modelRoiColumns: columns, topK: 1, onCompareModeChange: vi.fn(),
};

describe('presentation cleanup preserves controls and scientific evidence', () => {
  it('keeps controlled inputs, all ranking options, and the original callbacks', () => {
    const props = controlsProps();
    const elements = nativeElements(ControlsBar(props));
    const search = elements.find((element) => element.type === 'input' && element.props.type === 'search')!;
    const selects = elements.filter((element) => element.type === 'select');
    const checkboxes = elements.filter((element) => element.type === 'input' && element.props.type === 'checkbox');
    const reset = elements.find((element) => element.type === 'button')!;

    expect(search.props.value).toBe('clip');
    expect(selects.map((element) => element.props.value)).toEqual(['roi', 'eba']);
    expect(checkboxes.map((element) => element.props.checked)).toEqual([true, false]);
    expect(elements.filter((element) => element.type === 'option').map((element) => element.props.value))
      .toEqual(['overall', 'roi', 'ffa', 'eba', 'ppa']);

    search.props.onChange!({ target: { value: 'resnet', checked: false } });
    selects[0].props.onChange!({ target: { value: 'overall', checked: false } });
    selects[1].props.onChange!({ target: { value: 'ppa', checked: false } });
    checkboxes[0].props.onChange!({ target: { value: '', checked: false } });
    checkboxes[1].props.onChange!({ target: { value: '', checked: true } });
    reset.props.onClick!();

    expect(props.onModelSearchChange).toHaveBeenCalledWith('resnet');
    expect(props.onRankingSystemChange).toHaveBeenCalledWith('overall');
    expect(props.onSelectedRankingRoiChange).toHaveBeenCalledWith('ppa');
    expect(props.onShowScoreLabelsChange).toHaveBeenCalledWith(false);
    expect(props.onCompareModeChange).toHaveBeenCalledWith(true);
    expect(props.onResetControls).toHaveBeenCalledOnce();
  });

  it('only shows comparison progress when comparison is active', () => {
    const props = controlsProps();
    const inactive = renderToStaticMarkup(<ControlsBar {...props} />);
    const active = renderToStaticMarkup(<ControlsBar {...props} compareMode />);

    expect(inactive).not.toContain('Compare off');
    expect(inactive).not.toContain('1 / 2 selected');
    expect(active).toContain('1 / 2 selected');
    expect(active).toContain('aria-live="polite"');
    expect(inactive).toContain('Compare selections');
  });

  it('keeps overall ranking usable without presenting an irrelevant ROI selector', () => {
    const props = controlsProps();
    const elements = nativeElements(ControlsBar({ ...props, rankingSystem: 'overall' }));

    expect(elements.filter((element) => element.type === 'select')).toHaveLength(1);
    expect(elements.filter((element) => element.type === 'option').map((element) => element.props.value))
      .toEqual(['overall', 'roi']);
    expect(elements.filter((element) => element.type === 'input' && element.props.type === 'checkbox')).toHaveLength(2);
  });

  it('gives one clear empty-state instruction without repeating selection or dataset counters', () => {
    const markup = renderToStaticMarkup(<DetailsPanel {...detailsProps} />);

    expect(markup).toContain('Choose a model &amp; region');
    expect(markup).toContain('Select a heatmap cell');
    expect(markup).not.toMatch(/Selection:?\s*None|Loaded images/i);
    expect(markup).not.toContain('Export JSON');
  });

  it('retains selected response values, image limits, actions, and collapsed source details', () => {
    const markup = renderToStaticMarkup(<DetailsPanel {...detailsProps} selectedCell={cells[0]} />);
    const source = closedDetails(markup, 'evidence-source-details');

    expect(markup).toContain('Alignment</dt><dd>0.800');
    expect(markup).toContain('ROI rank</dt><dd>#1');
    expect(markup).toContain('Overall alignment</dt><dd>0.750');
    expect(markup).toContain('Export JSON');
    expect(markup).toContain('Copy metadata');
    expect(markup).toContain('Images per group');
    for (const limit of [6, 9, 12]) expect(markup).toContain(`value="${limit}"`);
    expect(markup).toContain('aria-label="Highest predicted responses"');
    expect(markup).toContain('aria-label="Lowest predicted responses"');
    for (const row of rows) expect(markup).toContain(`alt="${row.image_name}"`);
    expect(source).toContain('Response details');
    expect(source).toContain('Highest response</dt><dd>4.000');
    expect(source).toContain('Lowest response</dt><dd>1.000');
    expect(source).toContain('clip_rn50_ffa');
    expect(source).toContain('mean predicted voxel responses, not model alignment scores');
  });

  it('opens comparison evidence only for an active comparison with two selections', () => {
    expect(renderToStaticMarkup(<CompareDrawer {...compareProps} compareMode={false} />)).toBe('');
    expect(renderToStaticMarkup(<CompareDrawer {...compareProps} compareCells={[cells[0]]} />)).toBe('');
    expect(renderToStaticMarkup(<CompareDrawer {...compareProps} />)).toContain('aria-label="Compare selections"');
  });

  it('preserves comparison metrics, JSON export, and image galleries while qualifying the percentage', () => {
    const markup = renderToStaticMarkup(<CompareDrawer {...compareProps} />);
    const metrics = closedDetails(markup, 'compare-metric-details');
    const images = closedDetails(markup, 'compare-image-details');

    expect(markup).toContain('Rank correlation (ρ): 0.000');
    expect(markup).toContain('Alignment Δ (A−B): +0.200');
    expect(markup).toContain('Export JSON');
    expect(metrics).toContain('Images ranked</dt><dd>4');
    expect(metrics).toContain('Mean response Δ (A−B)</dt><dd>+0.000');
    expect(metrics).toContain('Rescaled rank correlation</dt><dd>50.0%');
    expect(metrics).toContain('(ρ + 1) / 2 × 100, not the percentage of matching images');
    for (const title of ['Shared highest', 'Highest · A only', 'Highest · B only', 'Shared lowest', 'Lowest · A only', 'Lowest · B only']) {
      expect(images).toContain(`aria-label="${title}"`);
    }
    for (const row of rows) expect(images).toContain(`alt="${row.image_name}"`);
  });

  it('keeps interpretation and keyboard guidance accessible in the collapsed quick guide', () => {
    const markup = renderToStaticMarkup(<HelpBox />);
    const guide = closedDetails(markup, 'help-box');

    expect(guide).toContain('Quick guide');
    expect(guide).toContain('rows are models, columns are brain regions');
    expect(guide).toContain('color shows model alignment');
    expect(guide).toContain('Image clusters');
    expect(guide).toContain('Compare selections');
    expect(guide).toContain('Enter');
    expect(guide).toContain('Escape');
  });
});
