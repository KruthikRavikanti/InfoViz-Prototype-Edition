import { renderToStaticMarkup } from 'react-dom/server';
import type { ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { BRAIN_REFERENCE_VERSION } from '../services/brainReferenceAssets';
import { rawSessionFixture } from '../test/sessionFixture';
import { BrainMapDialog } from './BrainMapDialog';

const dialogState = vi.hoisted(() => ({ open: false }));

vi.mock('../generated/referenceCoverage', async (importOriginal) => {
  const original = await importOriginal<typeof import('../generated/referenceCoverage')>();
  return { ...original, REFERENCE_COVERAGE: { ...original.REFERENCE_COVERAGE, stochasticModels: ['vone_alexnet'] } };
});

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
  raw.comparison.reference_version = BRAIN_REFERENCE_VERSION;
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
  it('discloses stochastic reference sampling inside the brain dialog', () => {
    dialogState.open = true;
    vi.stubGlobal('document', { body: {} });
    const markup = renderToStaticMarkup(<BrainMapDialog selectedModel="vone_alexnet" />);
    expect(markup).toContain('reference predictions are a seeded sample; repeated predictions may vary');
  });
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
    expect(markup).toContain('Preparing shared response scale');
    expect(markup).not.toContain('model units · fixed scale');
    expect(markup).toContain('aria-busy="true"');
    expect(markup).toContain('Model predictions, not measured BOLD signals for these images.');
    expect(markup).toContain('<details class="brain-map-mapping-details">');
    expect(markup).toContain('<summary>Mapping details</summary>');
    expect(markup).toContain('line-nearest-neighbor mapper');
    expect(markup).toContain('pooled NSD subjects');
    expect(markup).toContain('Three EBA responses are unmapped and not plotted.');
    expect(markup).toContain('Color between source voxels is not anatomical activation.');
  });

  it('offers the verified reference brain map without an uploaded session', () => {
    const markup = renderToStaticMarkup(<BrainMapDialog selectedModel="clip_rn50" />);
    expect(markup).toContain('3D Brain Map');
    expect(markup).toContain('aria-haspopup="dialog"');
    expect(markup).toContain('aria-expanded="false"');
  });

  it('uses one bounded searchable selector instead of growing a wrapping button list for many uploads', () => {
    dialogState.open = true;
    vi.stubGlobal('document', { body: {} });
    const session = brainSession();
    const stimulus = session.stimuli[0];
    const result = session.brainMapResults!.stimuli[0];
    session.stimuli = Array.from({ length: 1000 }, (_, index) => ({
      ...stimulus, id: `upload-${index}`, displayName: `upload-${index}.png`,
    }));
    session.brainMapResults!.stimuli = session.stimuli.map((item) => ({ ...result, stimulusId: item.id }));
    const markup = renderToStaticMarkup(
      <BrainMapDialog session={session} selectedModel="clip_rn50" focusedStimulusId="upload-500" onFocusStimulus={vi.fn()} />,
    );
    expect(markup).toContain('Search brain-map images');
    expect(markup).toContain('Brain-map image');
    expect(markup).toContain('Uploads (1000)');
    expect(markup).toContain('1000 images');
    expect(markup).toContain('data-image-id="upload-500"');
    expect(markup).toContain('data-image-source="uploaded"');
    expect(markup).toContain('Previous brain-map image');
    expect(markup).toContain('Next brain-map image');
    expect(markup).toContain('Murty-185 reference (185)');
    expect(markup).not.toContain('brain-map-stimulus-options');
    // Only navigation and existing display controls are buttons; image names stay in native options.
    expect(markup.match(/<button /g)).toHaveLength(5);
  });

  it('does not offer uploads from another model as reference voxel predictions', () => {
    dialogState.open = true;
    vi.stubGlobal('document', { body: {} });
    const markup = renderToStaticMarkup(
      <BrainMapDialog session={brainSession()} selectedModel="alexnet" focusedStimulusId="stimulus-a" />,
    );
    expect(markup).toContain('Uploads (0)');
    expect(markup).toContain('data-image-source="reference"');
    expect(markup).toContain('Select clip_rn50 to view this session');
    expect(markup).not.toContain('data-image-id="stimulus-a"');
    expect(markup).toContain('Loading reference predictions');
  });

  it('rejects incompatible upload bundles immediately, even before reference assets finish loading', () => {
    dialogState.open = true;
    vi.stubGlobal('document', { body: {} });
    const session = brainSession();
    session.comparison.referenceVersion = 'incompatible-bundle';
    const markup = renderToStaticMarkup(
      <BrainMapDialog session={session} selectedModel="clip_rn50" focusedStimulusId="stimulus-a" />,
    );
    expect(markup).toContain('Uploads (0)');
    expect(markup).toContain('Uploads use a different reference bundle');
    expect(markup).not.toContain('data-image-id="stimulus-a"');
  });
});
