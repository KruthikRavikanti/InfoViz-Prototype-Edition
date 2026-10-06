import type { CSSProperties } from 'react';
import type { CortexComparisonSession } from '../types/session';
import { CANONICAL_ROIS } from '../types/session';
import { BrainMapDialog } from './BrainMapDialog';

type SessionBannerProps = {
  session: CortexComparisonSession;
  focusedStimulusId: string | null;
  onFocusStimulus: (stimulusId: string) => void;
};

function formatValue(value: number): string {
  return value.toFixed(3);
}

export function SessionBanner({ session, focusedStimulusId, onFocusStimulus }: SessionBannerProps) {
  const focusedId = focusedStimulusId ?? session.stimuli[0]?.id;
  const uploadLabel = `${session.stimuli.length} upload${session.stimuli.length === 1 ? '' : 's'}`;

  return (
    <section className="session-context" aria-label="Cortex comparison session">
      <div className="session-context-status">
        <span className="session-status-indicator" aria-hidden="true" />
        <div>
          <strong>Comparison active</strong>
          <span>{session.comparison.model} · NSD-1000 · FFA, PPA, EBA</span>
        </div>
      </div>

      <div className="session-stimuli" aria-label="Your stimuli">
        <span className="session-stimuli-label">{uploadLabel}</span>
        <div className="session-stimuli-list">
          {session.stimuli.map((stimulus) => (
            <button
              key={stimulus.id}
              type="button"
              className={`session-stimulus ${stimulus.id === focusedId ? 'active' : ''}`}
              style={{ '--stimulus-color': stimulus.color } as CSSProperties}
              onClick={() => onFocusStimulus(stimulus.id)}
              aria-pressed={stimulus.id === focusedId}
            >
              <img src={stimulus.imageUrl} alt="" />
              <span>{stimulus.displayName}</span>
            </button>
          ))}
        </div>
      </div>

      <BrainMapDialog
        session={session}
        focusedStimulusId={focusedStimulusId}
        onFocusStimulus={onFocusStimulus}
      />

      <details className="session-context-details">
        <summary>Comparison details</summary>
        <div className="session-context-detail-body">
          <p className="session-context-copy">
            The original result used <strong>{session.sourceRequest.model}</strong> with{' '}
            <strong>{session.sourceRequest.dataset}</strong> / <strong>{session.sourceRequest.roi.toUpperCase()}</strong>.{' '}
            This fixed-reference comparison uses <strong>{session.comparison.model}</strong> with NSD-1000 predictions
            across FFA, PPA, and EBA. It does not change the original Cortex result.
          </p>

          <div className="session-roi-comparison" aria-label="Uploaded stimulus comparison across ROIs">
            <div className="session-stimuli-heading">
              <strong>Mean predicted response by region</strong>
              <span>NSD-1000 mapping</span>
            </div>
            <div className="session-roi-table-wrap">
              <table className="session-roi-table">
                <thead>
                  <tr>
                    <th>Stimulus</th>
                    {CANONICAL_ROIS.map((roi) => <th key={roi}>{roi.toUpperCase()}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {session.stimuli.map((stimulus) => (
                    <tr key={stimulus.id} className={stimulus.id === focusedId ? 'focused' : ''}>
                      <th>
                        <button type="button" onClick={() => onFocusStimulus(stimulus.id)}>
                          <span className="stimulus-dot" style={{ backgroundColor: stimulus.color }} />
                          {stimulus.displayName}
                        </button>
                      </th>
                      {CANONICAL_ROIS.map((roi) => {
                        const result = session.roiResults[roi].stimuli.find((item) => item.stimulusId === stimulus.id)!;
                        return (
                          <td key={roi}>
                            <strong>{formatValue(result.mean)}</strong>
                            <span>#{result.sessionRank}/{result.sessionCount} · {result.referencePercentile.toFixed(1)}th pct</span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="session-units">Predictions only; no measured fMRI for uploads.</p>
          </div>
        </div>
      </details>
    </section>
  );
}
