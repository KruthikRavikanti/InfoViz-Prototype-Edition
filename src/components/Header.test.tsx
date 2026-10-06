import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Header } from './Header';

function renderHeader() {
  return renderToStaticMarkup(<Header imageCount={185} modelCount={26} roiCount={3} />);
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('Cortex visualization masthead', () => {
  it('uses same-origin navigation and a base-aware local logo by default', () => {
    vi.stubEnv('VITE_CORTEX_APP_URL', undefined);
    vi.stubEnv('BASE_URL', '/infoviz/');

    const markup = renderHeader();

    expect(markup).toContain('class="cortex-brand" href="/" aria-label="Virtual Visual Cortex home"');
    expect(markup).toContain('src="/infoviz/cortex-logo.svg" alt=""');
    expect(markup).toContain('<span>Cortex</span>');
    expect(markup).toContain('aria-label="Cortex navigation"');
    expect(markup).toContain('<span class="app-product-label">Visualization</span>');
    expect(markup).toContain('class="back-to-lab" href="/lab/"');
    expect(markup).toContain('Back to the Lab');
  });

  it.each([
    ['https://cortex.example', 'https://cortex.example/', 'https://cortex.example/lab/'],
    [' https://cortex.example/research/cortex/ ', 'https://cortex.example/research/cortex/', 'https://cortex.example/research/cortex/lab/'],
    ['/research/cortex///', '/research/cortex/', '/research/cortex/lab/'],
  ])('preserves the configured Cortex deployment path: %s', (configured, home, lab) => {
    vi.stubEnv('VITE_CORTEX_APP_URL', configured);

    const markup = renderHeader();

    expect(markup).toContain(`class="cortex-brand" href="${home}"`);
    expect(markup).toContain(`class="back-to-lab" href="${lab}"`);
  });

  it('falls back to same-origin links for an empty configuration', () => {
    vi.stubEnv('VITE_CORTEX_APP_URL', '  ');

    const markup = renderHeader();

    expect(markup).toContain('class="cortex-brand" href="/"');
    expect(markup).toContain('class="back-to-lab" href="/lab/"');
  });

  it('describes the model and stimulus explorer with unchanged reference counts', () => {
    const markup = renderHeader();

    expect(markup).toContain('class="app-header-content"');
    expect(markup).toContain('<h1>Model &amp; Stimulus Explorer</h1>');
    expect(markup).toContain('Compare model alignment and explore image responses');
    expect(markup).toContain('aria-label="Dataset summary"');
    expect(markup).toContain('<dt>References</dt><dd>185</dd>');
    expect(markup).toContain('<dt>Models</dt><dd>26</dd>');
    expect(markup).toContain('<dt>ROIs</dt><dd>3</dd>');
  });
});
