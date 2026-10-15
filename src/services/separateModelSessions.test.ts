import { afterEach, describe, expect, it, vi } from 'vitest';
import { extractSessionToken, loadCortexComparisonSession } from './cortexSession';
import { rawJointHierarchyFixture, rawMultiModelSessionFixture, rawSessionFixture } from '../test/sessionFixture';
import { SUBJECT_ONE_ROI_VOXEL_COUNTS, CANONICAL_ROIS } from '../types/session';
import { getSessionForModel, getSessionModels } from '../utils/sessionModels';

const tokens = ['primary_token_123456', 'efficient_token_1234', 'clip_token_12345678', 'vgg_token_123456789'];
const models = ['alexnet', 'efficient_net', 'clip_rn50', 'vgg16_imagenet1kv1'];
const bundledToken = `multi-v1.${tokens.join('.')}`;
const png = (pixel: number) => new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, pixel]);

function rawSeparateSession(index: number) {
  const raw = rawSessionFixture();
  const ids = [`model-${index}-image-a`, `model-${index}-image-b`];
  return {
    ...raw,
    source_request: { model: 'alexnet', dataset: 'murty185', roi: 'eba' },
    comparison: { ...raw.comparison, model: models[index] },
    expires_at: `2099-01-01T0${4 - index}:00:00Z`,
    stimuli: ids.map((id, imageIndex) => ({
      ...raw.stimuli[0], id, display_name: imageIndex === 0 ? 'shared-name.png' : 'other.png',
      image_url: `/comparison-sessions/${tokens[index]}/images/${id}`,
    })),
    roi_results: Object.fromEntries(CANONICAL_ROIS.map((roi) => [roi, {
      ...raw.roi_results[roi],
      stimuli: ids.map((id, imageIndex) => ({
        ...raw.roi_results[roi].stimuli[0], stimulus_id: id, mean: index + imageIndex / 10,
        session_count: 187, session_rank: 6 + imageIndex,
      })),
      joint_hierarchy: rawJointHierarchyFixture(ids),
    }])),
    brain_map_results: {
      schema_version: 1, subject_id: 's1', response_space: 'nsd_1000_subject1_floc_t7_v1',
      roi_voxel_counts: { ...SUBJECT_ONE_ROI_VOXEL_COUNTS },
      stimuli: ids.map((id, imageIndex) => ({
        stimulus_id: id,
        roi_responses: Object.fromEntries(CANONICAL_ROIS.map((roi) => [roi,
          Array.from({ length: SUBJECT_ONE_ROI_VOXEL_COUNTS[roi] }, () => index + imageIndex / 10)])),
      })),
    },
    visual_results: {
      stimuli: ids.map((id) => ({ stimulus_id: id,
        projection: { x: 1, y: 2, cluster_id: 0, centroid_distance: 0.1, nearest_reference_members: [] },
        hierarchical_assignment: null })),
      joint_hierarchy: rawJointHierarchyFixture(ids),
    },
    dreamsim_results: {
      stimuli: ids.map((id) => ({ stimulus_id: id,
        projection: { x: 1, y: 2, cluster_id: 0, centroid_distance: 0.1, nearest_reference_members: [] },
        hierarchical_assignment: null })),
    },
  };
}

function bundleFetcher(sessions = models.map((_, index) => rawSeparateSession(index))) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const tokenIndex = tokens.findIndex((token) => url.includes(`/comparison-sessions/${token}`));
    if (tokenIndex < 0) throw new Error(`Unexpected fetch: ${url}`);
    if (url.includes('/images/')) {
      return new Response(png(url.endsWith('-image-b') ? 2 : 1), { headers: { 'Content-Type': 'image/png' } });
    }
    return new Response(JSON.stringify(sessions[tokenIndex]), { headers: { 'Content-Type': 'application/json' } });
  });
}

afterEach(() => vi.unstubAllGlobals());

describe('separate single-model session transport', () => {
  it('accepts two to four unique opaque tokens and keeps legacy tokens compatible', () => {
    expect(extractSessionToken(`#session=${bundledToken}`)).toBe(bundledToken);
    expect(extractSessionToken(`#session=multi-v1.${tokens.slice(0, 2).join('.')}`)).toBe(`multi-v1.${tokens.slice(0, 2).join('.')}`);
    expect(extractSessionToken(`#session=${tokens[0]}`)).toBe(tokens[0]);
  });

  it.each([
    'multi-v1.', `multi-v1.${tokens[0]}`, `multi-v1.${tokens[0]}.${tokens[0]}`,
    `multi-v1.${tokens.join('.')}.extra_token_12345678`, `multi-v1.${tokens[0]}.../../path`,
    `multi-v2.${tokens.slice(0, 2).join('.')}`, `multi-v1.${tokens[0]}.short`,
    `multi-v1.${tokens[0]}.${'x'.repeat(257)}`,
  ])('rejects malformed or duplicated handoffs before fetching: %s', async (token) => {
    const fetcher = bundleFetcher();
    expect(() => extractSessionToken(`#session=${token}`)).toThrow(/malformed/i);
    await expect(loadCortexComparisonSession(token, fetcher)).rejects.toThrow(/malformed/i);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('assembles four actual model payloads and remaps every companion upload identifier', async () => {
    const fetcher = bundleFetcher();
    const session = await loadCortexComparisonSession(bundledToken, fetcher);
    expect(getSessionModels(session)).toEqual(models);
    expect(session.sourceRequest).toEqual({ model: 'alexnet', dataset: 'murty185', roi: 'eba' });
    expect(session.expiresAt).toBe('2099-01-01T01:00:00Z');
    expect(session.stimuli.map((stimulus) => stimulus.id)).toEqual(['model-0-image-a', 'model-0-image-b']);
    expect(session.visualResults?.jointHierarchy?.members[185].id).toBe('model-0-image-a');
    expect(session.dreamsimResults?.stimuli[1].stimulusId).toBe('model-0-image-b');
    for (const [index, model] of models.entries()) {
      const scoped = getSessionForModel(session, model)!;
      for (const roi of CANONICAL_ROIS) {
        expect(scoped.roiResults[roi].stimuli.map((stimulus) => stimulus.stimulusId)).toEqual(session.stimuli.map((stimulus) => stimulus.id));
        expect(scoped.roiResults[roi].stimuli[0].mean).toBe(index);
        expect(scoped.roiResults[roi].jointHierarchy?.members[185].id).toBe('model-0-image-a');
        expect(scoped.roiResults[roi].jointHierarchy?.members[186].id).toBe('model-0-image-b');
      }
      expect(scoped.brainMapResults?.stimuli[0].roiResponses.ffa[0]).toBe(index);
      expect(scoped.brainMapResults?.stimuli.map((stimulus) => stimulus.stimulusId)).toEqual(session.stimuli.map((stimulus) => stimulus.id));
    }
    expect(fetcher).toHaveBeenCalledTimes(12); // Four sessions and eight verified image assets.
    expect(fetcher.mock.calls.some(([url]) => String(url).includes('multi-v1.'))).toBe(false);
  });

  it('does not require a backend model_results field and handles three deduplicated baseline sessions', async () => {
    const session = await loadCortexComparisonSession(`multi-v1.${tokens.slice(1).join('.')}`, bundleFetcher());
    expect(session.comparison.model).toBe('efficient_net');
    expect(getSessionModels(session)).toEqual(models.slice(1));
  });

  it('does not fetch identity images for single legacy tokens or old combined payloads', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify(rawMultiModelSessionFixture())));
    const session = await loadCortexComparisonSession(tokens[0], fetcher);
    expect(session.modelResults?.efficient_net.roiResults.ffa.stimuli[0].mean).toBe(-0.1);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('does not substitute a primary brain map when a companion has no brain vectors', async () => {
    const sessions = models.map((_, index) => rawSeparateSession(index));
    delete (sessions[1] as { brain_map_results?: unknown }).brain_map_results;
    const session = await loadCortexComparisonSession(bundledToken, bundleFetcher(sessions));
    expect(getSessionForModel(session, 'efficient_net')?.brainMapResults).toBeUndefined();
    expect(session.brainMapResults).toBeDefined();
  });

  const incompatible: [string, (sessions: ReturnType<typeof rawSeparateSession>[]) => void, RegExp][] = [
    ['duplicate models', (sessions) => { sessions[1].comparison.model = 'alexnet'; }, /duplicate models/i],
    ['source model', (sessions) => { sessions[1].source_request.model = 'clip_rn50'; }, /original Lab settings/i],
    ['source dataset', (sessions) => { sessions[1].source_request.dataset = 'nsd_1000'; }, /original Lab settings/i],
    ['source ROI', (sessions) => { sessions[1].source_request.roi = 'ffa'; }, /original Lab settings/i],
    ['reference version', (sessions) => { sessions[1].comparison.reference_version = 'different'; }, /different historical reference/i],
    ['reference count', (sessions) => { sessions[1].comparison.reference_count = 184; }, /185-image reference/i],
    ['missing ROI', (sessions) => { sessions[1].comparison.rois.pop(); }, /exactly FFA/i],
    ['image count', (sessions) => { sessions[1].stimuli.pop(); }, /uploaded image counts/i],
    ['display name', (sessions) => { sessions[1].stimuli[0].display_name = 'another.png'; }, /same image names/i],
    ['display-name order', (sessions) => { sessions[1].stimuli.reverse(); }, /uploaded stimulus order/i],
    ['ROI result order', (sessions) => { sessions[1].roi_results.ffa.stimuli.reverse(); }, /uploaded stimulus order/i],
    ['brain result order', (sessions) => { sessions[1].brain_map_results.stimuli.reverse(); }, /uploaded stimulus order/i],
    ['visual result order', (sessions) => { sessions[1].visual_results.stimuli.reverse(); }, /uploaded stimulus order/i],
    ['DreamSim result order', (sessions) => { sessions[1].dreamsim_results.stimuli.reverse(); }, /uploaded stimulus order/i],
    ['response values', (sessions) => { sessions[1].roi_results.ffa.stimuli[0].mean = Number.NaN; }, /finite/i],
    ['ROI reference order', (sessions) => {
      for (const roi of CANONICAL_ROIS) {
        const members = sessions[1].roi_results[roi].joint_hierarchy.members;
        [members[0].id, members[1].id] = [members[1].id, members[0].id];
      }
    }, /preserve reference image order/i],
    ['reference identity', (sessions) => {
      for (const roi of CANONICAL_ROIS) sessions[1].roi_results[roi].joint_hierarchy.members[0].id = 'wrong-reference.png';
      sessions[1].visual_results.joint_hierarchy.members[0].id = 'wrong-reference.png';
    }, /preserve reference image order|canonical reference image IDs/i],
    ['expired companion', (sessions) => { sessions[1].expires_at = '2020-01-01T01:00:00Z'; }, /expired/i],
  ];

  it.each(incompatible)('fails closed for mismatched %s before image downloads', async (_, mutate, message) => {
    const sessions = models.map((_, index) => rawSeparateSession(index));
    mutate(sessions);
    const fetcher = bundleFetcher(sessions);
    await expect(loadCortexComparisonSession(bundledToken, fetcher)).rejects.toThrow(message);
    expect(fetcher.mock.calls.some(([url]) => String(url).includes('/images/'))).toBe(false);
  });

  it('rejects nested old combined sessions inside a new token bundle', async () => {
    const sessions = models.map((_, index) => rawSeparateSession(index));
    Object.assign(sessions[1], { model_results: {} });
    // Empty legacy maps normalize away; nonempty valid maps must fail closed.
    Object.assign(sessions[1], { model_results: { another_model: {
      roi_results: sessions[1].roi_results, brain_map_results: sessions[1].brain_map_results,
    } } });
    await expect(loadCortexComparisonSession(bundledToken, bundleFetcher(sessions))).rejects.toThrow(/single-model sessions only/i);
  });

  it('allows differing visual-feature leaf order without changing the canonical reference set', async () => {
    const sessions = models.map((_, index) => rawSeparateSession(index));
    const members = sessions[1].visual_results.joint_hierarchy.members;
    [members[0].id, members[1].id] = [members[1].id, members[0].id];
    await expect(loadCortexComparisonSession(bundledToken, bundleFetcher(sessions))).resolves.toMatchObject({ comparison: { model: 'alexnet' } });
  });

  it('rejects different pixels despite matching display names and metadata', async () => {
    const regular = bundleFetcher();
    const fetcher = vi.fn(async (input: RequestInfo | URL) => String(input).includes(`${tokens[1]}/images/`)
      ? new Response(png(9)) : regular(input));
    await expect(loadCortexComparisonSession(bundledToken, fetcher)).rejects.toThrow(/different images or image order/i);
  });

  it('verifies pixels in upload order even when two images have the same display name', async () => {
    const sessions = models.map((_, index) => rawSeparateSession(index));
    for (const session of sessions) session.stimuli[1].display_name = session.stimuli[0].display_name;
    const regular = bundleFetcher(sessions);
    const fetcher = vi.fn(async (input: RequestInfo | URL) => String(input).includes(`${tokens[1]}/images/`)
      ? new Response(png(String(input).endsWith('-image-b') ? 1 : 2)) : regular(input));
    await expect(loadCortexComparisonSession(bundledToken, fetcher)).rejects.toThrow(/different images or image order/i);
  });

  it.each([404, 410, 503])('fails closed when an identity image is unavailable (HTTP %s)', async (status) => {
    const regular = bundleFetcher();
    const fetcher = vi.fn(async (input: RequestInfo | URL) => String(input).includes(`${tokens[1]}/images/`)
      ? new Response('', { status }) : regular(input));
    await expect(loadCortexComparisonSession(bundledToken, fetcher)).rejects.toMatchObject({ kind: status === 410 ? 'expired' : 'unavailable' });
  });

  it('requires valid sanitized PNG assets and secure hashing', async () => {
    const regular = bundleFetcher();
    const malformed = vi.fn(async (input: RequestInfo | URL) => String(input).includes('/images/')
      ? new Response('<html>missing image</html>') : regular(input));
    await expect(loadCortexComparisonSession(bundledToken, malformed)).rejects.toThrow(/invalid comparison image/i);
    vi.stubGlobal('crypto', {});
    await expect(loadCortexComparisonSession(bundledToken, regular)).rejects.toMatchObject({ kind: 'configuration' });
  });

  it('bounds simultaneous identity image downloads to four', async () => {
    const regular = bundleFetcher();
    let active = 0;
    let maximum = 0;
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      if (!String(input).includes('/images/')) return regular(input);
      active += 1;
      maximum = Math.max(maximum, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return regular(input);
    });
    await loadCortexComparisonSession(bundledToken, fetcher);
    expect(maximum).toBe(4);
    expect(active).toBe(0);
  });

  it('cancels in-flight image checks and stops draining the queue on failure', async () => {
    const regular = bundleFetcher();
    const fetcher = vi.fn(async (input: RequestInfo | URL, _options?: RequestInit) => {
      if (!String(input).includes('/images/')) return regular(input);
      if (String(input).endsWith(`${tokens[0]}/images/model-0-image-a`)) return new Response('', { status: 404 });
      await new Promise((resolve) => setTimeout(resolve, 5));
      return regular(input);
    });
    await expect(loadCortexComparisonSession(bundledToken, fetcher)).rejects.toMatchObject({ kind: 'unavailable' });
    const imageCalls = fetcher.mock.calls.filter(([input]) => String(input).includes('/images/'));
    expect(imageCalls.length).toBeLessThanOrEqual(4);
    expect(imageCalls.every(([, options]) => options?.signal?.aborted)).toBe(true);
  });

  it('rechecks the earliest expiration after downloading identity assets', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.parse('2099-01-01T01:00:01Z'));
    try {
      await expect(loadCortexComparisonSession(bundledToken, bundleFetcher())).rejects.toMatchObject({ kind: 'expired' });
    } finally {
      clock.mockRestore();
    }
  });

  it.each([404, 410, 500])('does not expose partial model results when a session GET fails (HTTP %s)', async (status) => {
    const regular = bundleFetcher();
    const fetcher = vi.fn(async (input: RequestInfo | URL) => String(input).endsWith(`/comparison-sessions/${tokens[1]}`)
      ? new Response('', { status }) : regular(input));
    await expect(loadCortexComparisonSession(bundledToken, fetcher)).rejects.toMatchObject({
      kind: status === 410 ? 'expired' : status === 404 ? 'invalid' : 'unavailable',
    });
    expect(fetcher.mock.calls.some(([url]) => String(url).includes('/images/'))).toBe(false);
  });
});
