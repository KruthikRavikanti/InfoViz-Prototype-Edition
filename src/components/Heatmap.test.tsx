import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { Heatmap } from './Heatmap';
import type { AggregateHeatmapCell } from '../types/data';
import { heatmapScoreColor } from '../utils/heatmapAppearance';

const cell: AggregateHeatmapCell = {
  id: 'ffa__clip_rn50',
  roi: 'ffa',
  model: 'clip_rn50',
  score: 0.812,
  rankWithinRoi: 4,
  overallScore: 0.75,
};

const defaults = {
  cells: [cell],
  models: ['clip_rn50'],
  rois: ['ffa'],
  showScoreLabels: true,
  sortDirection: 'desc' as const,
  selectedCell: null,
  compareMode: false,
  compareCells: [],
  comparableCellIds: new Set([cell.id]),
  sessionCellIds: new Set<string>(),
  onSortDirectionToggle: vi.fn(),
  onSelectCell: vi.fn(),
};

describe('session heatmap highlighting', () => {
  it('marks an existing cell without changing its model score', () => {
    const markup = renderToStaticMarkup(
      <Heatmap
        cells={[cell]}
        models={['clip_rn50']}
        rois={['ffa']}
        showScoreLabels
        sortDirection="desc"
        selectedCell={null}
        compareMode={false}
        compareCells={[]}
        comparableCellIds={new Set([cell.id])}
        sessionCellIds={new Set([cell.id])}
        onSortDirectionToggle={vi.fn()}
        onSelectCell={vi.fn()}
      />,
    );

    expect(markup).toContain('session-evidence');
    expect(markup).toContain('data-session-evidence="true"');
    expect(markup).toContain('data-score="0.812"');
    expect(cell.score).toBe(0.812);
  });
});

describe('compact heatmap presentation', () => {
  it('uses the scoreboard palette and a single compact legend with the real full score domain', () => {
    const markup = renderToStaticMarkup(<Heatmap {...defaults} colorDomain={[-0.281, 0.86]} />);
    expect(markup.match(/class="heatmap-color-key"/g)).toHaveLength(1);
    expect(markup).toContain('#ede9d8 0%, #c4b4cc 50%, #8966a3 100%');
    expect(markup).toContain('Alignment score color scale from -0.281 to 0.860');
    expect(markup).toContain(`fill="${heatmapScoreColor(cell.score!, [-0.281, 0.86])}"`);
    expect(markup).not.toContain('heatmap-legend');
  });

  it('retains compact square cell dimensions and score labels when only one model matches', () => {
    const markup = renderToStaticMarkup(<Heatmap {...defaults} />);
    const cellAttributes = markup.match(/<rect class="heatmap-cell"[^>]+/)?.[0];
    expect(cellAttributes).toContain('width="24"');
    expect(cellAttributes).toContain('height="24"');
    expect(markup).toContain('class="cell-score-label"');
    expect(markup).toContain('>0.81</text>');
    expect(markup).toContain('score 0.812');
    expect(markup).toContain('rank 4');
  });

  it('shows score labels in all three compact ROI columns', () => {
    const cells = ['ffa', 'ppa', 'eba'].map((roi) => ({ ...cell, roi, id: `${roi}__clip_rn50` }));
    const markup = renderToStaticMarkup(<Heatmap {...defaults} cells={cells} rois={['ffa', 'ppa', 'eba']} />);
    expect(markup.match(/class="cell-score-label"/g)).toHaveLength(3);
  });

  it('preserves score-toggle, selection, ranking highlight, and keyboard affordances', () => {
    const markup = renderToStaticMarkup(
      <Heatmap {...defaults} selectedCell={cell} selectedRankingRoi="ffa" showScoreLabels={false} sortDirection="asc" />,
    );
    expect(markup).toContain('heatmap-cell selected');
    expect(markup).toContain('class="roi-column-highlight"');
    expect(markup).toContain('aria-label="Sort ascending — click to toggle"');
    expect(markup).toContain('role="button" tabindex="0"');
    expect(markup).toContain('Press Enter to select.');
    expect(markup).not.toContain('class="cell-score-label"');
  });

  it('preserves compare selections and explains unavailable image-level comparisons', () => {
    const unavailable = { ...cell, id: 'ffa__alexnet_random', model: 'alexnet_random', score: null };
    const markup = renderToStaticMarkup(
      <Heatmap {...defaults} cells={[cell, unavailable]} models={['clip_rn50', 'alexnet_random']} compareMode compareCells={[cell]} />,
    );
    expect(markup).toContain('heatmap-cell compare-selected');
    expect(markup).toContain('class="compare-cell-marker"');
    expect(markup).toContain('>A</text>');
    expect(markup).toContain('heatmap-cell compare-disabled');
    expect(markup).toContain('aria-disabled="true"');
    expect(markup).toContain('No image-level data for comparison.');
    expect(markup).toContain('Press Enter to toggle compare selection.');
    expect(markup).toContain('Not comparable');
    expect(markup).toContain('score N/A');
  });

  it('keeps the existing no-models empty state', () => {
    const markup = renderToStaticMarkup(<Heatmap {...defaults} cells={[]} models={[]} />);
    expect(markup).toContain('No models match');
    expect(markup).not.toContain('class="heatmap-svg"');
  });
});
