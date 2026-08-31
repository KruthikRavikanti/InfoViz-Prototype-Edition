import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { SessionEvidencePanel } from './SessionEvidencePanel';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { buildSessionEvidenceView } from '../utils/sessionEvidence';

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
    expect(markup).toContain('Markers outside the fixed historical range are pinned to the boundary.');
  });
});
