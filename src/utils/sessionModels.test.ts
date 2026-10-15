import { describe, expect, it } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { rawMultiModelSessionFixture, rawSessionFixture } from '../test/sessionFixture';
import type { BrainMapResults } from '../types/session';
import { getSessionForModel, getSessionModels } from './sessionModels';

describe('model-specific uploaded predictions', () => {
  it('keeps the Lab selection first and includes each prepared model once', () => {
    const session = validateComparisonSession(rawMultiModelSessionFixture());
    expect(getSessionModels(session)).toEqual([
      'alexnet', 'efficient_net', 'clip_rn50', 'vgg16_imagenet1kv1',
    ]);
    expect(getSessionForModel(session, 'alexnet')).toBe(session);
  });

  it('returns each model’s actual predictions with shared image identity and source settings', () => {
    const session = validateComparisonSession(rawMultiModelSessionFixture());
    for (const [index, model] of getSessionModels(session).slice(1).entries()) {
      const scoped = getSessionForModel(session, model);
      expect(scoped?.comparison.model).toBe(model);
      expect(scoped?.roiResults.ffa.stimuli[0].mean).toBe(-(index + 1) / 10);
      expect(scoped?.stimuli).toBe(session.stimuli);
      expect(scoped?.sourceRequest).toBe(session.sourceRequest);
      expect(scoped?.comparison.referenceVersion).toBe(session.comparison.referenceVersion);
      expect(scoped?.modelResults).toBeUndefined();
    }
    expect(session.comparison.model).toBe('alexnet');
    expect(session.roiResults.ffa.stimuli[0].mean).toBe(0.8);
  });

  it('never lends the primary brain responses to a model lacking an anatomical audit', () => {
    const session = validateComparisonSession(rawMultiModelSessionFixture());
    const brain: BrainMapResults = {
      schemaVersion: 1,
      subjectId: 's1',
      responseSpace: 'nsd_1000_subject1_floc_t7_v1',
      roiVoxelCounts: { ffa: 156, ppa: 170, eba: 518 },
      stimuli: [],
    };
    session.brainMapResults = brain;
    session.modelResults!.clip_rn50.brainMapResults = { ...brain, stimuli: [] };
    expect(getSessionForModel(session, 'efficient_net')?.brainMapResults).toBeUndefined();
    expect(getSessionForModel(session, 'clip_rn50')?.brainMapResults).toBe(session.modelResults!.clip_rn50.brainMapResults);
    expect(getSessionForModel(session, 'alexnet')?.brainMapResults).toBe(brain);
  });

  it('shares model-independent visual features without borrowing response patterns', () => {
    const session = validateComparisonSession(rawMultiModelSessionFixture());
    session.visualResults = { stimuli: [] };
    session.dreamsimResults = { stimuli: [] };
    const scoped = getSessionForModel(session, 'efficient_net');
    expect(scoped?.visualResults).toBe(session.visualResults);
    expect(scoped?.dreamsimResults).toBe(session.dreamsimResults);
    expect(scoped?.roiResults).not.toBe(session.roiResults);
  });

  it('continues to support legacy sessions and rejects models not evaluated on these uploads', () => {
    const session = validateComparisonSession(rawSessionFixture());
    expect(getSessionModels(session)).toEqual(['clip_rn50']);
    expect(getSessionForModel(session, 'efficient_net')).toBeNull();
    expect(getSessionForModel(null, 'clip_rn50')).toBeNull();
    expect(getSessionModels(undefined)).toEqual([]);
  });
});
