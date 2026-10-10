import { REFERENCE_COVERAGE } from '../generated/referenceCoverage';

export const STOCHASTIC_REFERENCE_NOTE = 'Stochastic model: reference predictions are a seeded sample; repeated predictions may vary.';

export function isStochasticReferenceModel(model: string): boolean {
  return REFERENCE_COVERAGE.stochasticModels.includes(model);
}
