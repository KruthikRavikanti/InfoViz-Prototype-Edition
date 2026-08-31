import type { CSSProperties } from 'react';
import type { CortexComparisonSession } from '../types/session';
import { CANONICAL_ROIS } from '../types/session';

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

  return (
    <section className="session-context" aria-label="Cortex comparison session">
      <div className="session-context-copy">
        <p className="eyebrow">Cortex comparison session</p>
        <h2>Your stimuli in the historical image context</h2>
        <p>
          Your original result used <strong>{session.sourceRequest.model}</strong> with{' '}
          <strong>{session.sourceRequest.dataset}</strong> / <strong>{session.sourceRequest.roi.toUpperCase()}</strong>.{' '}
          {session.sourceRequest.model === session.comparison.model ? (
            <>This separate comparison uses that same <strong>{session.comparison.model}</strong> model</>
          ) : (
            <>You explicitly selected <strong>{session.comparison.model}</strong> as the supported comparison model</>
          )}{' '}
          with <strong>NSD-1000 predictions</strong> across FFA, PPA, and EBA. It does not change the original Cortex result.
        </p>
      </div>

      <div className="session-stimuli" aria-label="Your stimuli">
        <div className="session-stimuli-heading">
          <strong>Your stimuli</strong>
          <span>{session.stimuli.length} temporary upload{session.stimuli.length === 1 ? '' : 's'}</span>
        </div>
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

      <div className="session-roi-comparison" aria-label="Uploaded stimulus comparison across ROIs">
        <div className="session-stimuli-heading">
          <strong>Predicted response across canonical ROIs</strong>
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
        <p className="session-units">Values are predicted response under the NSD-1000 mapping; uploaded stimuli have no measured fMRI.</p>
      </div>
    </section>
  );
}
