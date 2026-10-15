import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { defaultSourceRoot } from './sync_scoreboard_scores.mjs';

const checkoutRoot = path.resolve('test-checkouts');
const scorePath = 'assets/data/new/standardized_results_nsd_1000_models_univariate.json';

describe('scoreboard source checkout discovery', () => {
  it('uses the containing Cortex App checkout for apps/cortex-compass', () => {
    const host = path.join(checkoutRoot, 'cortex-app');
    const child = path.join(host, 'apps/cortex-compass');
    const files = new Set([path.join(host, 'package.json'), path.join(host, scorePath)]);
    expect(defaultSourceRoot(child, (filename) => files.has(filename))).toBe(host);
  });

  it('retains the sibling cortex-app default for an independent checkout', () => {
    const child = path.join(checkoutRoot, 'InfoViz-Prototype-Edition');
    const exists = vi.fn(() => false);
    expect(defaultSourceRoot(child, exists)).toBe(path.join(checkoutRoot, 'cortex-app'));
    expect(exists).toHaveBeenCalledWith(path.resolve(child, '../../package.json'));
  });

  it('does not mistake a containing package without the exact score file for the host', () => {
    const child = path.join(checkoutRoot, 'cortex-app/apps/cortex-compass');
    const host = path.resolve(child, '../..');
    expect(defaultSourceRoot(child, (filename) => filename === path.join(host, 'package.json')))
      .toBe(path.resolve(child, '../cortex-app'));
  });

  it('requires a containing host package even when a matching score path exists', () => {
    const child = path.join(checkoutRoot, 'cortex-app/apps/cortex-compass');
    const host = path.resolve(child, '../..');
    expect(defaultSourceRoot(child, (filename) => filename === path.join(host, scorePath)))
      .toBe(path.resolve(child, '../cortex-app'));
  });
});
