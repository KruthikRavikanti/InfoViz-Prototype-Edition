import type { CSSProperties } from 'react';
import type { CortexComparisonSession } from '../types/session';
import { CANONICAL_ROIS } from '../types/session';
import { getSessionForModel, getSessionModels } from '../utils/sessionModels';

type SessionBannerProps = {
  session: CortexComparisonSession;
  focusedStimulusId: string | null;
  onFocusStimulus: (stimulusId: string) => void;
  selectedModel?: string | null;
  onSelectModel?: (model: string) => void;
};

function formatValue(value: number): string {
  return value.toFixed(3);
}

function modelLabel(model: string): string {
  if (model === 'efficient_net') return 'EfficientNet';
  if (model === 'clip_rn50') return 'CLIP RN50';
  if (model === 'vgg16_imagenet1kv1') return 'VGG16 (ImageNet)';
  return model;
}

export function SessionBanner({ session, focusedStimulusId, onFocusStimulus, selectedModel, onSelectModel }: SessionBannerProps) {
  const focusedId = focusedStimulusId ?? session.stimuli[0]?.id;
  const uploadLabel = `${session.stimuli.length} upload${session.stimuli.length === 1 ? '' : 's'}`;
  const models = getSessionModels(session);
  const activeSession = getSessionForModel(session, selectedModel ?? session.comparison.model);
  const sourceDataset = session.sourceRequest.dataset === 'nsd_1000' ? 'NSD-1000'
    : session.sourceRequest.dataset === 'murty185' ? 'Murty-185' : session.sourceRequest.dataset;

  return (
    <section className="session-context" aria-label="Cortex-Compass comparison session">
      <div className="session-context-status">
        <span className="session-status-indicator" aria-hidden="true" />
        <div>
          <strong>Comparison active</strong>
          <span>{activeSession
            ? `${activeSession.comparison.model} · NSD-1000 · FFA, PPA, EBA`
            : `${selectedModel} · reference images only`}</span>
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

      {models.length > 1 && (
        <div className="session-model-controls" aria-label="Models prepared for your stimuli">
          <span>{models.length} models</span>
          {models.map((model) => (
            <button
              key={model}
              type="button"
              aria-pressed={activeSession?.comparison.model === model}
              aria-label={`View uploaded stimuli with ${model}`}
              title={`${model}${model === session.sourceRequest.model ? ' · Lab selection' : ''}`}
              disabled={!onSelectModel}
              onClick={() => onSelectModel?.(model)}
            >{modelLabel(model)}</button>
          ))}
        </div>
      )}

      <details className="session-context-details">
        <summary>Comparison details</summary>
        <div className="session-context-detail-body">
          <p className="session-context-copy">
            Lab result: <strong>{session.sourceRequest.model}</strong> ·{' '}
            <strong>{sourceDataset}</strong> · <strong>{session.sourceRequest.roi.toUpperCase()}</strong>.
            {' '}Upload comparison: <strong>{models.join(', ')}</strong> · NSD-1000 · FFA, PPA, EBA.
            {' '}Your original Lab result is unchanged.
          </p>

          <div className="session-roi-comparison" aria-label="Uploaded stimulus comparison across ROIs">
            <div className="session-stimuli-heading">
              <strong>Mean predicted response by region</strong>
              <span>{activeSession ? `${activeSession.comparison.model} · NSD-1000 mapping` : 'Select a prepared model above'}</span>
            </div>
            {activeSession ? <div className="session-roi-table-wrap">
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
                        const result = activeSession.roiResults[roi].stimuli.find((item) => item.stimulusId === stimulus.id)!;
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
            </div> : <p>Uploads are not predicted for this model. Choose a prepared model to view them.</p>}
            <p className="session-units">Predictions only; no measured fMRI for uploads.</p>
          </div>
        </div>
      </details>
    </section>
  );
}
