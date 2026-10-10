import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';
import { SessionBanner } from './SessionBanner';

describe('comparison session context', () => {
  it('keeps uploads in a compact summary with technical details collapsed', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const markup = renderToStaticMarkup(
      <SessionBanner session={session} focusedStimulusId="stimulus-a" onFocusStimulus={vi.fn()} />,
    );

    expect(markup).toContain('aria-label="Cortex-Compass comparison session"');
    expect(markup).toContain('Comparison active');
    expect(markup).toContain('1 upload');
    expect(markup).toContain('uploaded.png');
    expect(markup).toContain('aria-pressed="true"');
    expect(markup).toContain('<details class="session-context-details">');
    expect(markup).toContain('Comparison details');
    expect(markup).toContain('Mean predicted response by region');
    expect(markup).toContain('Predictions only; no measured fMRI for uploads.');
    expect(markup).toContain('Your original Lab result is unchanged.');
    expect(markup).toContain('Lab result:');
    expect(markup).toContain('Upload comparison:');
    expect(markup).not.toContain('<h2>');
  });
});
