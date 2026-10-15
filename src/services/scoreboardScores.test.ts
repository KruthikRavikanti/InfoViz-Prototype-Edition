import { json } from 'd3';
import { afterEach, describe, expect, it, vi } from 'vitest';
import snapshot from '../../public/data/scoreboard/standardized_results_nsd_1000_models_univariate.json';
import snapshotText from '../../public/data/scoreboard/standardized_results_nsd_1000_models_univariate.json?raw';
import sourceManifest from '../../public/data/scoreboard/source.json';
import historicalScores from '../../public/data/murty185_zero_index.json';
import {
  CORTEX_SCOREBOARD_PATH,
  loadScoreboardScores,
  parseScoreboardScores,
  SCOREBOARD_SNAPSHOT_PATH,
} from './scoreboardScores';

vi.mock('d3', () => ({ json: vi.fn() }));

function fixture() {
  return {
    ffa: { alpha: { murty185: [0.8, 1] }, beta: { bmd_2024: [0.2, 0.3] } },
    ppa: { alpha: { murty185: [0.9, 1] }, beta: { murty185: [-0.2, -0.25] } },
    eba: { alpha: { murty185: [0.6, 0.7] }, beta: { murty185: [0.1, 0.125] } },
  };
}

afterEach(() => {
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

describe('Virtual Cortex scoreboard score source', () => {
  it('separates exact raw scores from saved normalized color values', () => {
    const result = parseScoreboardScores(fixture());
    expect(result.scores.ppa.alpha).toBe(0.9);
    expect(result.normalizedScores.ppa.alpha).toBe(1);
    expect(result.scores.ppa.beta).toBe(-0.2);
    expect(result.normalizedScores.ppa.beta).toBe(-0.25);
    // Normalized entries are stored benchmark values, not raw scores divided by a new ceiling.
    expect(result.normalizedScores.eba.alpha).toBe(0.7);
    expect(result.scores.Overall.alpha).toBe((0.8 + 0.9 + 0.6) / 3);
    expect(result.normalizedScores.Overall.alpha).toBe((1 + 1 + 0.7) / 3);
  });

  it('omits missing evaluations and averages only available ROI scores', () => {
    const { scores, normalizedScores } = parseScoreboardScores(fixture());
    expect(Object.hasOwn(scores.ffa, 'beta')).toBe(false);
    expect(Object.hasOwn(normalizedScores.ffa, 'beta')).toBe(false);
    expect(scores.Overall.beta).toBe((-0.2 + 0.1) / 2);
    expect(normalizedScores.Overall.beta).toBe((-0.25 + 0.125) / 2);
  });

  it('rejects an empty or wrong-evaluation source rather than showing an empty alignment table', () => {
    expect(() => parseScoreboardScores({ ffa: {}, ppa: {}, eba: {} })).toThrow(/no Murty185 evaluations/);
    expect(() => parseScoreboardScores({
      ffa: { alpha: { nsd_1000: [0.8, 1] } }, ppa: {}, eba: {},
    })).toThrow(/no Murty185 evaluations/);
  });

  it.each([null, undefined, [], 2, 'wrong'])('rejects invalid root data: %s', (raw) => {
    expect(() => parseScoreboardScores(raw)).toThrow(/Scoreboard data must be an object/);
  });

  it.each(['ffa', 'ppa', 'eba'])('rejects a missing or malformed %s ROI', (roi) => {
    const raw = fixture() as Record<string, unknown>;
    delete raw[roi];
    expect(() => parseScoreboardScores(raw)).toThrow(new RegExp(`${roi.toUpperCase()} data`));
    raw[roi] = [];
    expect(() => parseScoreboardScores(raw)).toThrow(new RegExp(`${roi.toUpperCase()} data`));
  });

  it.each([null, [], 0, 'bad'])('rejects malformed model evaluation data: %s', (value) => {
    const raw = fixture() as { ffa: Record<string, unknown> };
    raw.ffa.alpha = value;
    expect(() => parseScoreboardScores(raw)).toThrow(/FFA \/ alpha evaluations/);
  });

  it.each([
    null, [], [0.5], [0.5, null], [0.5, '0.7'], ['0.5', 0.7],
    [Number.NaN, 0.7], [0.5, Number.POSITIVE_INFINITY], [0.5, 0.7, 0.8],
  ])('rejects present scores without a finite raw/normalized pair: %j', (pair) => {
    const raw = fixture() as { ffa: { alpha: { murty185: unknown } } };
    raw.ffa.alpha.murty185 = pair;
    expect(() => parseScoreboardScores(raw)).toThrow(/FFA \/ alpha \/ Murty185 must contain two finite numbers/);
  });

  it('rejects sparse pairs rather than accepting missing normalized data', () => {
    const raw = fixture();
    raw.ffa.alpha.murty185 = [0.5, ,] as number[];
    expect(() => parseScoreboardScores(raw)).toThrow(/two finite numbers/);
  });

  it('does not let model names modify object prototypes', () => {
    const raw = fixture();
    Object.defineProperty(raw.ffa, '__proto__', { enumerable: true, value: { murty185: [0.3, 0.4] } });
    const { scores } = parseScoreboardScores(raw);
    expect(Object.hasOwn(scores.ffa, '__proto__')).toBe(true);
    expect(Object.getPrototypeOf(scores.ffa)).toBe(Object.prototype);
    expect(scores.Overall.__proto__).toBe(0.3);
  });

  it('loads the reviewed standalone snapshot when no host frame is active', async () => {
    vi.stubGlobal('window', undefined);
    vi.mocked(json).mockResolvedValueOnce(fixture());
    const result = await loadScoreboardScores();
    expect(json).toHaveBeenCalledExactlyOnceWith(SCOREBOARD_SNAPSHOT_PATH);
    expect(result.scores.ffa.alpha).toBe(0.8);
  });

  it('reads the host exact file while embedded in Virtual Cortex', async () => {
    vi.stubGlobal('window', { location: { search: '?embedded=1' }, parent: {}, self: {} });
    vi.mocked(json).mockResolvedValueOnce(fixture());
    await loadScoreboardScores();
    expect(json).toHaveBeenCalledExactlyOnceWith(CORTEX_SCOREBOARD_PATH);
  });

  it('does not treat the embedded query flag in a top-level window as a host', async () => {
    const environment = { location: { search: '?embedded=1' }, parent: undefined as unknown };
    environment.parent = environment;
    vi.stubGlobal('window', environment);
    vi.mocked(json).mockResolvedValueOnce(fixture());
    await loadScoreboardScores();
    expect(json).toHaveBeenCalledExactlyOnceWith(SCOREBOARD_SNAPSHOT_PATH);
  });

  it('supports an explicit score source without a hidden fallback', async () => {
    vi.mocked(json).mockResolvedValueOnce(fixture());
    await loadScoreboardScores('/reviewed/scores.json');
    expect(json).toHaveBeenCalledExactlyOnceWith('/reviewed/scores.json');
  });

  it('surfaces an unavailable host source without silently loading a stale snapshot', async () => {
    vi.stubGlobal('window', { location: { search: '?embedded=1' }, parent: {}, self: {} });
    vi.mocked(json).mockRejectedValueOnce(new Error('404 Not Found'));
    await expect(loadScoreboardScores()).rejects.toThrow(/Virtual Cortex scoreboard scores.*404 Not Found/);
    expect(json).toHaveBeenCalledExactlyOnceWith(CORTEX_SCOREBOARD_PATH);
  });

  it('surfaces malformed authoritative data without fallback', async () => {
    vi.mocked(json).mockResolvedValueOnce({ ffa: {} });
    await expect(loadScoreboardScores()).rejects.toThrow(/Virtual Cortex scoreboard scores.*PPA data/);
    expect(json).toHaveBeenCalledTimes(1);
  });

  it('retains all 381 historical raw cells alongside the added pixels baseline and exact source colors', () => {
    const parsed = parseScoreboardScores(snapshot);
    let count = 0;
    let historicalCount = 0;
    for (const roi of ['ffa', 'ppa', 'eba'] as const) {
      expect(Object.keys(parsed.scores[roi])).toHaveLength(128);
      for (const [model, evaluations] of Object.entries(snapshot[roi])) {
        count += 1;
        expect(parsed.scores[roi][model]).toBe(evaluations.murty185[0]);
        expect(parsed.normalizedScores[roi][model]).toBe(evaluations.murty185[1]);
      }
      for (const [model, historicalScore] of Object.entries(historicalScores[roi])) {
        historicalCount += 1;
        expect(parsed.scores[roi][model]).toBe(historicalScore);
      }
    }
    expect(count).toBe(384);
    expect(historicalCount).toBe(381);
    for (const [model, historicalScore] of Object.entries(historicalScores.Overall)) {
      expect(parsed.scores.Overall[model]).toBe(historicalScore);
    }
    expect(parsed.normalizedScores.ffa.blip2).toBe(0.996465);
    expect(parsed.normalizedScores.ppa.alexnet_random).toBe(-0.25736);
    expect(parsed.normalizedScores.eba.google_vit_random).toBe(-0.332436);
  });

  it('includes the upstream pixels baseline without modifying its raw or normalized scores', () => {
    const parsed = parseScoreboardScores(snapshot);
    expect(parsed.scores.ffa.pixels).toBe(-0.013275);
    expect(parsed.scores.ppa.pixels).toBe(0.106638);
    expect(parsed.scores.eba.pixels).toBe(-0.081218);
    expect(parsed.normalizedScores.ffa.pixels).toBe(-0.016186);
    expect(parsed.normalizedScores.ppa.pixels).toBe(0.130989);
    expect(parsed.normalizedScores.eba.pixels).toBe(-0.096097);
    expect(parsed.scores.Overall.pixels).toBe((-0.013275 + 0.106638 - 0.081218) / 3);
  });

  it('pins provenance to the exact source-file bytes and explicit scientific context', async () => {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(snapshotText));
    const hex = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
    expect(hex).toBe(sourceManifest.sourceSha256);
    expect(sourceManifest.sourceCommit).toMatch(/^[a-f0-9]{40}$/);
    expect(sourceManifest.sourcePath).toBe(CORTEX_SCOREBOARD_PATH.slice(1));
    expect(sourceManifest.trainingDataset).toBe('nsd_1000');
    expect(sourceManifest.evaluationDataset).toBe('murty185');
    expect(sourceManifest.metric).toBe('Univariate Pearson correlation');
    expect(sourceManifest.modelCount).toBe(127);
  });
});
