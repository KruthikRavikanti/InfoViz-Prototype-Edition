import type { AggregateHeatmapCell, JsonScoreObject, RankingSystem, Roi, SortDirection } from '../types/data';

export const DEFAULT_HEATMAP_ROIS: Roi[] = ['ffa', 'ppa', 'eba'];

function getScore(scores: JsonScoreObject, roi: Roi, model: string): number | null {
  const value = scores[roi]?.[model];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function compareScores(
  aScore: number | null,
  bScore: number | null,
  aModel: string,
  bModel: string,
  direction: SortDirection,
): number {
  if (aScore === null && bScore === null) {
    return aModel.localeCompare(bModel);
  }

  if (aScore === null) {
    return 1;
  }

  if (bScore === null) {
    return -1;
  }

  const difference = direction === 'desc' ? bScore - aScore : aScore - bScore;
  return difference || aModel.localeCompare(bModel);
}

export function getDefaultHeatmapRois(scores: JsonScoreObject): Roi[] {
  const availableRois = new Set(Object.keys(scores));
  const defaults = DEFAULT_HEATMAP_ROIS.filter((roi) => availableRois.has(roi));

  if (defaults.length > 0) {
    return defaults;
  }

  return Object.keys(scores).filter((roi) => roi !== 'Overall');
}

export function inferAggregateModels(scores: JsonScoreObject, rois: Roi[]): string[] {
  const modelNames = new Set<string>();

  for (const roi of [...rois, 'Overall']) {
    for (const model of Object.keys(scores[roi] ?? {})) {
      modelNames.add(model);
    }
  }

  return Array.from(modelNames);
}

export function orderAggregateModels(
  models: string[],
  scores: JsonScoreObject,
  rankingSystem: RankingSystem,
  selectedRoi: Roi,
  sortDirection: SortDirection,
): string[] {
  const sortedModels = [...models];
  const scoreRoi = rankingSystem === 'roi' ? selectedRoi : 'Overall';

  return sortedModels.sort((a, b) =>
    // The noise ceiling is a benchmark, not a competing model. Keep its row
    // separate from either direction of model ranking, as on the scoreboard.
    a === b ? 0 : a === 'ceiling' ? -1 : b === 'ceiling' ? 1
      : compareScores(getScore(scores, scoreRoi, a), getScore(scores, scoreRoi, b), a, b, sortDirection),
  );
}

export function filterModelsBySearch(models: string[], searchTerm: string): string[] {
  const normalizedSearch = searchTerm.trim().toLowerCase();

  if (!normalizedSearch) {
    return models;
  }

  return models.filter((model) => model.toLowerCase().includes(normalizedSearch));
}

export function rankModelsWithinRoi(scores: JsonScoreObject, roi: Roi): Map<string, number> {
  const scoredModels = Object.entries(scores[roi] ?? {})
    .filter((entry): entry is [string, number] => entry[0] !== 'ceiling' && typeof entry[1] === 'number' && Number.isFinite(entry[1]))
    .sort(([aModel, aScore], [bModel, bScore]) => bScore - aScore || aModel.localeCompare(bModel));

  return new Map(scoredModels.map(([model], index) => [model, index + 1]));
}

export function buildAggregateHeatmapCells(
  scores: JsonScoreObject,
  rois: Roi[],
  models: string[],
  normalizedScores?: JsonScoreObject,
): AggregateHeatmapCell[] {
  const rankMaps = new Map(rois.map((roi) => [roi, rankModelsWithinRoi(scores, roi)]));

  return rois.flatMap((roi) =>
    models.map((model) => {
      const score = getScore(scores, roi, model);

      return {
        id: `${roi}__${model}`,
        roi,
        model,
        score,
        normalizedScore: normalizedScores ? getScore(normalizedScores, roi, model) : null,
        rankWithinRoi: rankMaps.get(roi)?.get(model) ?? null,
        overallScore: getScore(scores, 'Overall', model),
      };
    }),
  );
}
