import { describe, expect, it } from 'vitest';
import { assertSessionReferenceCompatible, validateComparisonSession, validateReferenceManifest } from './cortexSession';
import { rawJointHierarchyFixture, rawMultiModelSessionFixture, rawReferenceManifestFixture } from '../test/sessionFixture';

describe('multi-model comparison session contract', () => {
  it('parses distinct companion responses without changing the primary model or uploads', () => {
    const raw = rawMultiModelSessionFixture();
    const session = validateComparisonSession(raw);
    expect(session.comparison.model).toBe('alexnet');
    expect(session.sourceRequest.model).toBe('alexnet');
    expect(session.modelResults?.efficient_net.roiResults.ffa.stimuli[0].mean).toBe(-0.1);
    expect(session.modelResults?.clip_rn50.roiResults.ffa.stimuli[0].mean).toBe(-0.2);
    expect(session.modelResults?.vgg16_imagenet1kv1.roiResults.ffa.stimuli[0].mean).toBe(-0.3);
    expect(session.roiResults.ffa.stimuli[0].mean).toBe(0.8);
  });

  it('rejects duplicated primary models, nested payloads, and more than three companions', () => {
    const raw = rawMultiModelSessionFixture();
    raw.model_results.alexnet = raw.model_results.efficient_net;
    expect(() => validateComparisonSession(raw)).toThrow(/at most three/i);
    delete raw.model_results.vgg16_imagenet1kv1;
    expect(() => validateComparisonSession(raw)).toThrow(/duplicate the primary/i);
    delete raw.model_results.alexnet;
    Object.assign(raw.model_results.efficient_net, { model_results: {} });
    expect(() => validateComparisonSession(raw)).toThrow(/unexpected fields/i);
  });

  it('validates every companion ROI, stimulus identity, and response value', () => {
    const raw = rawMultiModelSessionFixture();
    raw.model_results.efficient_net.roi_results.ffa.stimuli[0].mean = Number.NaN;
    expect(() => validateComparisonSession(raw)).toThrow(/finite/i);
    raw.model_results.efficient_net.roi_results.ffa.stimuli[0].mean = 0.2;
    raw.model_results.efficient_net.roi_results.ppa.stimuli[0].stimulus_id = 'another-upload';
    expect(() => validateComparisonSession(raw)).toThrow(/every uploaded stimulus/i);
    raw.model_results.efficient_net.roi_results.ppa.stimuli[0].stimulus_id = 'stimulus-a';
    Object.assign(raw.model_results.efficient_net.roi_results, { other: {} });
    expect(() => validateComparisonSession(raw)).toThrow(/exactly FFA, PPA, and EBA/i);
  });

  it('requires companion joint-clustering leaves to use the same reference image order', () => {
    const raw = rawMultiModelSessionFixture();
    Object.assign(raw.roi_results.ffa, { joint_hierarchy: rawJointHierarchyFixture() });
    const companionHierarchy = rawJointHierarchyFixture();
    [companionHierarchy.members[0].id, companionHierarchy.members[1].id] = [
      companionHierarchy.members[1].id, companionHierarchy.members[0].id,
    ];
    Object.assign(raw.model_results.efficient_net.roi_results.ffa, { joint_hierarchy: companionHierarchy });
    expect(() => validateComparisonSession(raw)).toThrow(/preserve reference image order/i);
  });

  it('checks reference support for all prepared models, not only the initial model', () => {
    const session = validateComparisonSession(rawMultiModelSessionFixture());
    const rawManifest = rawReferenceManifestFixture();
    rawManifest.models = ['alexnet', 'efficient_net', 'clip_rn50'];
    expect(() => assertSessionReferenceCompatible(session, validateReferenceManifest(rawManifest))).toThrow(/does not support/i);
    rawManifest.models.push('vgg16_imagenet1kv1');
    expect(() => assertSessionReferenceCompatible(session, validateReferenceManifest(rawManifest))).not.toThrow();
  });

  it('checks reference order between companions even when the primary has no joint tree', () => {
    const raw = rawMultiModelSessionFixture();
    Object.assign(raw.model_results.efficient_net.roi_results.ffa, { joint_hierarchy: rawJointHierarchyFixture() });
    const hierarchy = rawJointHierarchyFixture();
    [hierarchy.members[0].id, hierarchy.members[1].id] = [hierarchy.members[1].id, hierarchy.members[0].id];
    Object.assign(raw.model_results.clip_rn50.roi_results.ffa, { joint_hierarchy: hierarchy });
    expect(() => validateComparisonSession(raw)).toThrow(/preserve reference image order/i);
  });
});
