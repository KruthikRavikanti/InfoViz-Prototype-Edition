import type { CortexComparisonSession } from '../types/session';

/** Available upload predictions, with the Lab's chosen comparison model first. */
export function getSessionModels(session: CortexComparisonSession | null | undefined): string[] {
  if (!session) return [];
  return [...new Set([session.comparison.model, ...Object.keys(session.modelResults ?? {})])];
}

/** Scope every response view to one model; never borrow another model's brain vectors. */
export function getSessionForModel(
  session: CortexComparisonSession | null | undefined,
  model: string,
): CortexComparisonSession | null {
  if (!session) return null;
  if (session.comparison.model === model) return session;
  if (!session.modelResults || !Object.hasOwn(session.modelResults, model)) return null;
  const results = session.modelResults[model];
  return {
    ...session,
    comparison: { ...session.comparison, model },
    roiResults: results.roiResults,
    // Explicitly override even when absent: a missing anatomical audit is not
    // permission to display the primary model's voxel predictions instead.
    brainMapResults: results.brainMapResults,
    // This slice is for a single model. Availability controls use the root
    // session, which retains all models and the original initial selection.
    modelResults: undefined,
  };
}
