import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { SessionEvidencePanel } from './SessionEvidencePanel';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { buildSessionEvidenceView, UPLOADED_STIMULUS_MARKER_COLOR } from '../utils/sessionEvidence';

describe('session response distribution', () => {
  it('visibly explains uploaded markers pinned outside the fixed reference domain', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    session.roiResults.ffa.stimuli[0].mean = 1000;
    const rows = Array.from({ length: 185 }, (_, index) => ({
      image_name: `image_${String(index + 1).padStart(3, '0')}.png`,
      clip_rn50_ffa: index / 100,
    }));
    const view = buildSessionEvidenceView(
      { id: 'ffa__clip_rn50', roi: 'ffa', model: 'clip_rn50', score: 0.8, rankWithinRoi: 1, overallScore: 0.7 },
      rows,
      [{ columnName: 'clip_rn50_ffa', model: 'clip_rn50', roi: 'ffa' }],
      session,
      'stimulus-a',
      6,
    );

    expect(view).not.toBeNull();
    const markup = renderToStaticMarkup(
      <SessionEvidencePanel view={view!} onOpenImage={vi.fn()} onFocusStimulus={vi.fn()} />,
    );
    expect(markup).toContain('Out-of-range uploads appear at the boundary.');
    expect(markup).toContain('185 Murty references + 1 upload');
    expect(markup).toContain('Axis fixed to the reference range.');
    expect(markup.match(/data-distribution-image="true"/g)).toHaveLength(186);
    expect(markup.match(/data-origin="reference"/g)).toHaveLength(185);
    expect(markup.match(/data-origin="uploaded"/g)).toHaveLength(1);
    expect(markup.match(/data-axis-y="73"/g)).toHaveLength(186);
    expect(markup.match(/distribution-reference-tick/g)).toHaveLength(185);
    expect(markup).toContain('data-distribution-interaction="nearest"');
    expect(markup).toContain('Use Left and Right Arrow keys to preview images and Enter to enlarge.');
    expect(markup).toContain('session-distribution-glow');
    expect(markup).toContain('session-distribution-upload');
    expect(markup.match(new RegExp(`fill="${UPLOADED_STIMULUS_MARKER_COLOR}"`, 'g'))).toHaveLength(2);
    expect(markup).not.toContain('class="distribution-reference"');
    expect(markup).not.toContain('M 0 -14 L 8 -4 L 0 6 L -8 -4 Z');
  });

  it('distinguishes voxel SEM and fixed top/bottom membership from the displayed image limit', () => {
    const raw = rawSessionFixture();
    raw.roi_results.ffa.stimuli[0].session_rank = 8;
    const session = validateComparisonSession(raw, 'http://core.example:5556');
    const rows = Array.from({ length: 185 }, (_, index) => ({
      image_name: `image_${String(index + 1).padStart(3, '0')}.png`,
      clip_rn50_ffa: 1 - index / 100,
    }));
    const view = buildSessionEvidenceView(
      { id: 'ffa__clip_rn50', roi: 'ffa', model: 'clip_rn50', score: 0.8, rankWithinRoi: 1, overallScore: 0.7 },
      rows,
      [{ columnName: 'clip_rn50_ffa', model: 'clip_rn50', roi: 'ffa' }],
      session,
      'stimulus-a',
      6,
    )!;
    const markup = renderToStaticMarkup(
      <SessionEvidencePanel view={view} onOpenImage={vi.fn()} onFocusStimulus={vi.fn()} />,
    );

    expect(view.topImages).toHaveLength(6);
    expect(view.topImages.some((image) => image.stimulusId === 'stimulus-a')).toBe(false);
    expect(markup).toContain('Voxel SEM');
    expect(markup).toContain('NSD-1000 predicted mean.');
    expect(markup).toContain('Descending insertion rank among 185 reference predictions; ties receive their midpoint');
    expect(markup).toContain('Reference rank</dt><dd>#5.5</dd>');
    expect(markup).not.toContain('relative to 185 references');
    expect(markup).toContain('Standard error across predicted voxel responses; not model uncertainty');
    expect(markup).toContain('Top / bottom 10');
    expect(markup).toContain('<dd>Top 10</dd>');
    expect(markup).not.toContain('Enters the displayed');
    expect(markup).toContain('Similar response patterns');
    expect(markup).toContain('<details class="session-neighbors">');
    expect(markup).toContain('<details class="session-full-ranking">');
  });

  it('keeps below-reference ranks explicit and only shows the boundary warning when needed', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const rows = Array.from({ length: 185 }, (_, index) => ({
      image_name: `image_${String(index + 1).padStart(3, '0')}.png`,
      clip_rn50_ffa: index / 100,
    }));
    const buildView = () => buildSessionEvidenceView(
      { id: 'ffa__clip_rn50', roi: 'ffa', model: 'clip_rn50', score: 0.8, rankWithinRoi: 1, overallScore: 0.7 },
      rows,
      [{ columnName: 'clip_rn50_ffa', model: 'clip_rn50', roi: 'ffa' }],
      session,
      'stimulus-a',
      6,
    )!;
    const markup = renderToStaticMarkup(
      <SessionEvidencePanel view={buildView()} onOpenImage={vi.fn()} onFocusStimulus={vi.fn()} />,
    );

    expect(markup).not.toContain('Out-of-range uploads appear at the boundary.');
    for (const label of ['Predicted mean', 'Voxel SEM', 'Reference rank', 'Reference percentile', 'Combined rank', 'Top / bottom 10']) {
      expect(markup).toContain(`>${label}</dt>`);
    }
    expect(markup).toContain('aria-label="Focus uploaded stimulus"');
    expect(markup).toContain('aria-label="Similar response patterns"');
    expect(markup).toContain('aria-label="Combined FFA ranking"');

    Object.assign(session.roiResults.ffa.stimuli[0], {
      mean: -1,
      referenceRank: 186,
      referencePercentile: 0,
      sessionRank: 186,
      entersTop: false,
      entersBottom: true,
    });
    const belowReferenceMarkup = renderToStaticMarkup(
      <SessionEvidencePanel view={buildView()} onOpenImage={vi.fn()} onFocusStimulus={vi.fn()} />,
    );
    expect(belowReferenceMarkup).toContain('Reference rank</dt><dd>Below all 185</dd>');
    expect(belowReferenceMarkup).toContain('Out-of-range uploads appear at the boundary.');
  });
});
