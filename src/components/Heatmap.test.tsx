import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { Heatmap } from './Heatmap';
import type { AggregateHeatmapCell } from '../types/data';

const cell: AggregateHeatmapCell = {
  id: 'ffa__clip_rn50',
  roi: 'ffa',
  model: 'clip_rn50',
  score: 0.812,
  rankWithinRoi: 4,
  overallScore: 0.75,
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
