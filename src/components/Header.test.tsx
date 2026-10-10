import pageHtml from '../../index.html?raw';
import type { ComponentProps } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import type { SelectedHeatmapCell } from '../types/data';
import { Header } from './Header';

function renderHeader(props: Partial<ComponentProps<typeof Header>> = {}) {
  return renderToStaticMarkup(<Header imageCount={185} modelCount={26} roiCount={3} {...props} />);
}

function visibleText(markup: string): string {
  return markup.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
}

function cell(model = 'clip_rn50', roi = 'ffa'): SelectedHeatmapCell {
  return {
    id: `${roi}__${model}`,
    model,
    roi,
    score: 0.8,
    rankWithinRoi: 1,
    overallScore: 0.75,
  };
}

function comparisonSession() {
  const raw = rawSessionFixture();
  raw.source_request.model = 'alexnet';
  return validateComparisonSession(raw, 'http://core.example:5556');
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('Cortex-Compass masthead', () => {
  it('uses same-origin navigation and a base-aware local logo by default', () => {
    vi.stubEnv('VITE_CORTEX_APP_URL', undefined);
    vi.stubEnv('BASE_URL', '/infoviz/');

    const markup = renderHeader();

    expect(markup).toContain('class="cortex-brand" href="/" aria-label="Virtual Visual Cortex home"');
    expect(markup).toContain('src="/infoviz/cortex-logo.svg" alt=""');
    expect(markup).toContain('<span>Cortex</span>');
    expect(markup).toContain('aria-label="Cortex navigation"');
    expect(markup).toContain('<span class="app-product-label">Cortex-Compass</span>');
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

  it('names Cortex-Compass and identifies its fixed reference collection and counts', () => {
    const markup = renderHeader();
    const text = visibleText(markup);

    expect(markup).toContain('class="app-header-content"');
    expect(markup).toContain('<h1>Cortex-Compass</h1>');
    expect(markup).not.toContain('Murty Lab · Vision AI');
    expect(markup).not.toContain('Model &amp; Stimulus Explorer');
    expect(text).toContain('Murty-185');
    expect(text).toContain('185 images');
    expect(text).toMatch(/select.*model.*region/i);
    expect(text).not.toContain('NSD-1000');
    expect(markup).toContain('aria-label="Dataset summary"');
    expect(markup).toContain('<dt>References</dt><dd>185</dd>');
    expect(markup).toContain('<dt>Models</dt><dd>26</dd>');
    expect(markup).toContain('<dt>ROIs</dt><dd>3</dd>');
  });

  it('identifies the active model, region, and NSD-1000 prediction mapping for image-backed cells', () => {
    const text = visibleText(renderHeader({ selectedCell: cell(), hasImageResponses: true }));

    expect(text).toContain('Murty-185');
    expect(text).toMatch(/Viewing:.*clip_rn50.*FFA/);
    expect(text).toMatch(/NSD-1000 (?:mapping|predictions)/);
    expect(text).not.toContain('alignment scores only');
    expect(text).not.toContain('Your Lab settings:');
  });

  it('does not claim an NSD-1000 mapping for a score-only model', () => {
    const text = visibleText(renderHeader({ selectedCell: cell('siglip2', 'ppa') }));

    expect(text).toMatch(/Viewing:.*siglip2.*PPA/);
    expect(text).toContain('alignment scores only');
    expect(text).not.toContain('NSD-1000');
  });

  it('keeps original Lab choices separate from the current upload-comparison mapping', () => {
    const text = visibleText(renderHeader({
      selectedCell: cell(),
      hasImageResponses: true,
      session: comparisonSession(),
    }));

    expect(text).toMatch(/Viewing:.*clip_rn50.*FFA.*NSD-1000/);
    expect(text).toMatch(/Your Lab settings:.*alexnet.*Murty-185.*EBA/);
    expect(text).not.toMatch(/Your Lab settings:.*alexnet.*NSD-1000/);
  });

  it('describes the loaded upload comparison before a heatmap cell is selected', () => {
    const text = visibleText(renderHeader({ session: comparisonSession() }));

    expect(text).toMatch(/Upload comparison:.*clip_rn50.*NSD-1000/);
    expect(text).toMatch(/FFA.*PPA.*EBA/);
    expect(text).toMatch(/Your Lab settings:.*alexnet.*Murty-185.*EBA/);
  });

  it('preserves the upload model identity when viewing another model', () => {
    const text = visibleText(renderHeader({
      selectedCell: cell('clip_rn101', 'ppa'),
      hasImageResponses: true,
      session: comparisonSession(),
    }));

    expect(text).toMatch(/Viewing:.*clip_rn101.*PPA/);
    expect(text).toMatch(/Uploads:.*clip_rn50.*NSD-1000/);
    expect(text).toMatch(/Your Lab settings:.*alexnet.*Murty-185.*EBA/);
  });

  it('names both comparison selections and does not imply uploads are part of the reference comparison', () => {
    const text = visibleText(renderHeader({
      compareMode: true,
      compareCells: [cell(), cell('clip_rn101', 'ppa')],
      session: comparisonSession(),
    }));

    expect(text).toMatch(/Reference comparison:.*clip_rn50.*FFA.*clip_rn101.*PPA/);
    expect(text).toContain('reference images only');
    expect(text).toMatch(/Uploads:.*clip_rn50.*NSD-1000/);
  });

  it('shows a selection instruction before either reference comparison cell is chosen', () => {
    const text = visibleText(renderHeader({ compareMode: true }));

    expect(text).toMatch(/choose.*two.*model.*region.*compare/i);
    expect(text).toContain('reference images only');
    expect(text).not.toContain('Viewing:');
  });

  it('identifies the first comparison selection and prompts for the second', () => {
    const text = visibleText(renderHeader({ compareMode: true, compareCells: [cell()] }));

    expect(text).toMatch(/Reference comparison:.*clip_rn50.*FFA/);
    expect(text).toMatch(/choose.*second.*cell/i);
    expect(text).toContain('reference images only');
    expect(text).not.toContain('Viewing:');
  });

  it('preserves unfamiliar original Lab setting identifiers instead of guessing their dataset meaning', () => {
    const session = comparisonSession();
    session.sourceRequest.dataset = 'custom_dataset_v2';
    session.sourceRequest.roi = 'earlyvisualcortex';
    const text = visibleText(renderHeader({ session }));

    expect(text).toMatch(/Your Lab settings:.*custom_dataset_v2.*EARLYVISUALCORTEX/);
    expect(text).not.toMatch(/custom_dataset_v2.*NSD-1000/);
  });

  it('uses the same product name in the browser title and page description', () => {
    expect(pageHtml).toContain('<title>Cortex-Compass — Virtual Cortex</title>');
    expect(pageHtml).toContain('name="description" content="Cortex-Compass compares model alignment');
    expect(pageHtml).not.toContain('Model &amp; Stimulus Explorer');
  });
});
