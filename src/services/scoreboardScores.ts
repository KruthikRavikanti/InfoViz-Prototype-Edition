import { json } from 'd3';
import type { JsonScoreObject } from '../types/data';
import { publicPath } from '../utils/publicPath';
import { isCortexEmbedded } from './cortexEmbedding';

export const CORTEX_SCOREBOARD_PATH = '/assets/data/new/standardized_results_nsd_1000_models_univariate.json';
export const SCOREBOARD_SNAPSHOT_PATH = publicPath('/data/scoreboard/standardized_results_nsd_1000_models_univariate.json');
const SCOREBOARD_ROIS = ['ffa', 'ppa', 'eba'] as const;
const EVALUATION_DATASET = 'murty185';

export type ScoreboardScores = {
  scores: JsonScoreObject;
  normalizedScores: JsonScoreObject;
};

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Scoreboard ${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

/** Keep benchmark correlations separate from the scoreboard's saved color values. */
export function parseScoreboardScores(raw: unknown): ScoreboardScores {
  const source = object(raw, 'data');
  const scores: JsonScoreObject = { ffa: {}, ppa: {}, eba: {}, Overall: {} };
  const normalizedScores: JsonScoreObject = { ffa: {}, ppa: {}, eba: {}, Overall: {} };
  const modelNames = new Set<string>();

  for (const roi of SCOREBOARD_ROIS) {
    const models = object(source[roi], `${roi.toUpperCase()} data`);
    const rawEntries: [string, number][] = [];
    const normalizedEntries: [string, number][] = [];

    for (const [model, value] of Object.entries(models)) {
      const evaluations = object(value, `${roi.toUpperCase()} / ${model} evaluations`);
      // Missing evaluation data are not a zero correlation or a zero color value.
      if (!Object.hasOwn(evaluations, EVALUATION_DATASET)) continue;
      const pair = evaluations[EVALUATION_DATASET];
      if (!Array.isArray(pair) || pair.length !== 2
        || typeof pair[0] !== 'number' || !Number.isFinite(pair[0])
        || typeof pair[1] !== 'number' || !Number.isFinite(pair[1])) {
        throw new Error(`Scoreboard ${roi.toUpperCase()} / ${model} / Murty185 must contain two finite numbers: raw and normalized scores.`);
      }
      rawEntries.push([model, pair[0]]);
      normalizedEntries.push([model, pair[1]]);
      modelNames.add(model);
    }
    scores[roi] = Object.fromEntries(rawEntries);
    normalizedScores[roi] = Object.fromEntries(normalizedEntries);
  }

  if (modelNames.size === 0) {
    throw new Error('Scoreboard data contain no Murty185 evaluations for FFA, PPA, or EBA.');
  }

  const overall = (values: JsonScoreObject): Record<string, number> => Object.fromEntries(
    Array.from(modelNames, (model) => {
      const available = SCOREBOARD_ROIS.map((roi) => values[roi][model]).filter(Number.isFinite);
      return [model, available.reduce((sum, score) => sum + score, 0) / available.length];
    }),
  );
  scores.Overall = overall(scores);
  normalizedScores.Overall = overall(normalizedScores);
  return { scores, normalizedScores };
}

/** Embedded Compass reads the host's exact score file; standalone uses its reviewed snapshot. */
export async function loadScoreboardScores(
  path = isCortexEmbedded() ? CORTEX_SCOREBOARD_PATH : SCOREBOARD_SNAPSHOT_PATH,
): Promise<ScoreboardScores> {
  try {
    return parseScoreboardScores(await json<unknown>(path));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    // A failed host request must not silently switch to a potentially stale snapshot.
    throw new Error(`Could not load Virtual Cortex scoreboard scores from ${path}: ${detail}`);
  }
}
