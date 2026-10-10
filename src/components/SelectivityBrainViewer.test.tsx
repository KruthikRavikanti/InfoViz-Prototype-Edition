import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import type { BrainMapAssets } from '../types/brainMap';
import SelectivityBrainViewer from './SelectivityBrainViewer';

describe('selectivity brain renderer shell', () => {
  it('keeps surface loading within the existing bounded brain canvas', () => {
    const markup = renderToStaticMarkup(
      <SelectivityBrainViewer
        assets={{} as BrainMapAssets}
        points={[]}
        enabledRois={new Set(['ffa', 'eba', 'ppa'])}
        surface="inflated"
        selectedCluster={null}
        resetKey={0}
        onHover={vi.fn()}
        onSelectCluster={vi.fn()}
      />,
    );
    expect(markup).toContain('brain-map-viewer-shell selectivity-brain-viewer-shell');
    expect(markup).toContain('class="brain-map-canvas"');
    expect(markup).toContain('role="status"');
    expect(markup).toContain('Loading inflated cortical surface');
    expect(markup).not.toContain('UMAP');
  });

  it('labels pial loading distinctly rather than implying similarity coordinates', () => {
    const markup = renderToStaticMarkup(
      <SelectivityBrainViewer
        assets={{} as BrainMapAssets}
        points={[]}
        enabledRois={new Set(['ffa'])}
        surface="pial"
        selectedCluster={2}
        resetKey={1}
        onHover={vi.fn()}
        onSelectCluster={vi.fn()}
      />,
    );
    expect(markup).toContain('Loading pial cortical surface');
    expect(markup).not.toContain('UMAP');
  });
});
