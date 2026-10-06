import { renderToStaticMarkup } from 'react-dom/server';
import type { ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { BrainMapDialog } from './BrainMapDialog';

const dialogState = vi.hoisted(() => ({ open: false }));

vi.mock('react', async (importOriginal) => {
  const original = await importOriginal<typeof import('react')>();
  return {
    ...original,
    useState(initialState: unknown) {
      return original.useState(dialogState.open && initialState === false ? true : initialState);
    },
  };
});

vi.mock('react-dom', async (importOriginal) => ({
  ...await importOriginal<typeof import('react-dom')>(),
  createPortal: (children: ReactNode) => children,
}));

afterEach(() => {
  dialogState.open = false;
  vi.unstubAllGlobals();
});

function brainSession() {
  const raw = rawSessionFixture() as ReturnType<typeof rawSessionFixture> & { brain_map_results?: unknown };
  raw.brain_map_results = {
    schema_version: 1,
    subject_id: 's1',
    response_space: 'nsd_1000_subject1_floc_t7_v1',
    roi_voxel_counts: { ffa: 156, ppa: 170, eba: 518 },
    stimuli: [{
      stimulus_id: 'stimulus-a',
      roi_responses: {
        ffa: Array.from({ length: 156 }, () => 0),
        ppa: Array.from({ length: 170 }, () => 0),
        eba: Array.from({ length: 518 }, () => 0),
      },
    }],
  };
  return validateComparisonSession(raw, 'http://core.example:5556');
}

describe('Subject 1 brain-map launcher', () => {
  it('keeps the 3D experience behind an accessible session-area dialog trigger', () => {
    const session = brainSession();
    const markup = renderToStaticMarkup(
      <BrainMapDialog
        session={session}
        focusedStimulusId="stimulus-a"
        onFocusStimulus={vi.fn()}
      />,
    );

    expect(markup).toContain('3D Brain Map');
    expect(markup).toContain('aria-haspopup="dialog"');
    expect(markup).toContain('aria-expanded="false"');
    expect(markup).not.toContain('role="dialog"');
  });

  it('keeps scientific provenance available in a collapsed mapping disclosure', () => {
    dialogState.open = true;
    vi.stubGlobal('document', { body: {} });
    const markup = renderToStaticMarkup(
      <BrainMapDialog
        session={brainSession()}
        focusedStimulusId="stimulus-a"
        onFocusStimulus={vi.fn()}
      />,
    );

    expect(markup).toContain('role="dialog"');
    expect(markup).toContain('Predicted responses · NSD Subject 1');
    expect(markup).toContain('One fixed scale across uploads and regions.');
    expect(markup).toContain('Model predictions, not measured BOLD signals for the uploaded image.');
    expect(markup).toContain('<details class="brain-map-mapping-details">');
    expect(markup).toContain('<summary>Mapping details</summary>');
    expect(markup).toContain('line-nearest-neighbor mapper');
    expect(markup).toContain('pooled NSD subjects');
    expect(markup).toContain('Three EBA responses are unmapped and not plotted.');
    expect(markup).toContain('Color between source voxels is not anatomical activation.');
  });
});
