import { describe, expect, it } from 'vitest';
import { csvParse } from 'd3';
import source from '../../public/data/scoreboard/standardized_results_nsd_1000_models_univariate.json';
import historicalScores from '../../public/data/murty185_zero_index.json';
import referenceCsv from '../../public/data/combined_image_means.csv?raw';
import { parseScoreboardScores } from '../services/scoreboardScores';
import { inferModelRoiColumns } from './data';
import { buildEvidenceView, hasComparableData } from './evidence';
import {
  buildAggregateHeatmapCells,
  filterModelsBySearch,
  inferAggregateModels,
  orderAggregateModels,
  rankModelsWithinRoi,
} from './heatmap';

const rois = ['ffa', 'ppa', 'eba'];
const { scores, normalizedScores } = parseScoreboardScores(source);
const models = inferAggregateModels(scores, rois);

describe('scoreboard-consistent model alignment', () => {
  it('preserves all historical raw correlations, not scores from a different training or evaluation dataset', () => {
    for (const roi of [...rois, 'Overall'] as const) {
      for (const [model, historicalScore] of Object.entries(historicalScores[roi as keyof typeof historicalScores])) {
        expect(scores[roi][model]).toBe(historicalScore);
      }
    }
    expect(models).toHaveLength(128);
    expect(models.filter((model) => model !== 'ceiling')).toHaveLength(127);
    const cells = buildAggregateHeatmapCells(scores, rois, models, normalizedScores);
    expect(cells).toHaveLength(384);
    expect(cells.filter((cell) => cell.model !== 'pixels')).toHaveLength(381);
    expect(cells.every((cell) => cell.score !== null && cell.normalizedScore !== null)).toBe(true);
  });

  it('ranks models but not the noise ceiling, including scores above the ceiling', () => {
    for (const roi of rois) expect(rankModelsWithinRoi(scores, roi).has('ceiling')).toBe(false);
    expect(rankModelsWithinRoi(scores, 'ffa').get('clip_rn50')).toBe(4);
    expect(rankModelsWithinRoi(scores, 'ppa').get('clip_rn50')).toBe(8);
    expect(rankModelsWithinRoi(scores, 'eba').get('clip_rn50')).toBe(43);
    expect(rankModelsWithinRoi(scores, 'eba').get('resnet50')).toBe(5);
    expect(scores.ffa.clip_rn50).toBeGreaterThan(scores.ffa.ceiling);
    expect(normalizedScores.ffa.clip_rn50).toBe(1);
    expect(buildAggregateHeatmapCells(scores, rois, ['ceiling'], normalizedScores)
      .every((cell) => cell.rankWithinRoi === null)).toBe(true);
  });

  it('keeps pixels score-only without borrowing another model\'s reference predictions', () => {
    const rows = csvParse(referenceCsv, (row) => ({ ...row, image_name: String(row.image_name ?? '') }));
    const columns = inferModelRoiColumns(rows, rois);
    expect(rows).toHaveLength(185);
    expect(columns).toHaveLength(378);
    const pixelsCells = buildAggregateHeatmapCells(scores, rois, ['pixels'], normalizedScores);
    expect(pixelsCells).toHaveLength(3);
    for (const cell of pixelsCells) {
      expect(cell.score).not.toBeNull();
      expect(cell.normalizedScore).not.toBeNull();
      expect(hasComparableData(cell.model, cell.roi, rows, columns)).toBe(false);
      expect(buildEvidenceView(cell, rows, columns, 6)).toBeNull();
      expect(hasComparableData('clip_rn50', cell.roi, rows, columns)).toBe(true);
    }
    expect(rankModelsWithinRoi(scores, 'ffa').get('pixels')).toBe(127);
    expect(rankModelsWithinRoi(scores, 'ppa').get('pixels')).toBe(121);
    expect(rankModelsWithinRoi(scores, 'eba').get('pixels')).toBe(126);
  });

  it('keeps the noise-ceiling reference first while sorting real models by raw scores', () => {
    for (const direction of ['asc', 'desc'] as const) {
      const ordered = orderAggregateModels(models, scores, 'roi', 'eba', direction);
      expect(ordered[0]).toBe('ceiling');
      for (let index = 2; index < ordered.length; index += 1) {
        const previous = scores.eba[ordered[index - 1]];
        const current = scores.eba[ordered[index]];
        if (direction === 'asc') expect(previous).toBeLessThanOrEqual(current);
        else expect(previous).toBeGreaterThanOrEqual(current);
      }
    }
  });

  it('uses unrounded Overall means for sorting and keeps full-roster ROI ranks after filtering', () => {
    const ordered = orderAggregateModels(models, scores, 'overall', 'ffa', 'desc');
    expect(ordered[1]).toBe('nomic');
    const filtered = filterModelsBySearch(ordered, 'clip_rn50');
    const cells = buildAggregateHeatmapCells(scores, rois, filtered, normalizedScores);
    expect(cells.find((cell) => cell.id === 'ffa__clip_rn50')).toMatchObject({
      score: 0.849183, normalizedScore: 1, rankWithinRoi: 4, overallScore: scores.Overall.clip_rn50,
    });
    expect(filterModelsBySearch(ordered, ' CEILING ')).toEqual(['ceiling']);
  });
});
