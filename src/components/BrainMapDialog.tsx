import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { loadBrainMapAssets } from '../services/brainMapAssets';
import type { BrainMapAssets, BrainMapHoverDatum, BrainSurfaceKind } from '../types/brainMap';
import {
  CANONICAL_ROIS,
  type CanonicalRoi,
  type CortexComparisonSession,
} from '../types/session';
import { fixedSessionResponseExtent } from '../utils/brainProjection';

const BrainMapViewer = lazy(() => import('./BrainMapViewer'));

const ROI_NAMES: Record<CanonicalRoi, string> = {
  ffa: 'Fusiform Face Area',
  ppa: 'Parahippocampal Place Area',
  eba: 'Extrastriate Body Area',
};

type AssetState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; assets: BrainMapAssets };

type BrainMapDialogProps = {
  session: CortexComparisonSession;
  focusedStimulusId: string | null;
  onFocusStimulus: (stimulusId: string) => void;
};

function formatResponse(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

function keepFocusInsideDialog(event: KeyboardEvent<HTMLElement>) {
  if (event.key !== 'Tab') return;
  const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(
    'button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href], summary, [tabindex]:not([tabindex="-1"])',
  )).filter((element) => element.getClientRects().length > 0);
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (!first || !last) return;
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function BrainMapModal({
  session,
  focusedStimulusId,
  onFocusStimulus,
  onClose,
}: BrainMapDialogProps & { onClose: () => void }) {
  const [assetState, setAssetState] = useState<AssetState>({ status: 'loading' });
  const [surface, setSurface] = useState<BrainSurfaceKind>('inflated');
  const [enabledRois, setEnabledRois] = useState<Record<CanonicalRoi, boolean>>({
    ffa: true,
    ppa: true,
    eba: true,
  });
  const [hovered, setHovered] = useState<BrainMapHoverDatum | null>(null);
  const [resetKey, setResetKey] = useState(0);

  const focusedId = focusedStimulusId ?? session.stimuli[0]?.id;
  const stimulus = session.stimuli.find((item) => item.id === focusedId) ?? session.stimuli[0];
  const stimulusResult = session.brainMapResults?.stimuli.find((item) => item.stimulusId === stimulus?.id)
    ?? session.brainMapResults?.stimuli[0];
  const enabledRoiSet = useMemo(
    () => new Set(CANONICAL_ROIS.filter((roi) => enabledRois[roi])),
    [enabledRois],
  );
  const maxAbsResponse = useMemo(
    () => fixedSessionResponseExtent(session.brainMapResults?.stimuli ?? []),
    [session.brainMapResults],
  );

  useEffect(() => {
    let ignore = false;
    const abortController = new AbortController();
    loadBrainMapAssets(fetch, abortController.signal)
      .then((assets) => {
        if (ignore) return;
        setAssetState({ status: 'ready', assets });
        setSurface(assets.manifest.surfaces.inflated ? 'inflated' : 'pial');
      })
      .catch((error: unknown) => {
        if (ignore) return;
        setAssetState({
          status: 'error',
          message: error instanceof Error ? error.message : 'The Subject 1 brain-map assets could not be loaded.',
        });
      });
    return () => {
      ignore = true;
      abortController.abort();
    };
  }, []);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previousOverflow; };
  }, []);

  const mappedCount = assetState.status === 'ready'
    ? CANONICAL_ROIS.reduce((sum, roi) => sum + assetState.assets.manifest.rois[roi].mappedResponseCount, 0)
    : null;
  const responseCount = session.brainMapResults
    ? CANONICAL_ROIS.reduce((sum, roi) => sum + session.brainMapResults!.roiVoxelCounts[roi], 0)
    : 0;

  return createPortal(
    <div
      className="brain-map-modal-backdrop"
      role="presentation"
      onClick={onClose}
      onKeyDown={(event) => {
        if (event.key === 'Escape') onClose();
      }}
    >
      <section
        className="brain-map-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="brain-map-modal-title"
        aria-describedby="brain-map-modal-description"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={keepFocusInsideDialog}
      >
        <header className="brain-map-modal-header">
          <div>
            <p className="eyebrow">Predicted responses · NSD Subject 1</p>
            <h2 id="brain-map-modal-title">3D Brain Map</h2>
            <p id="brain-map-modal-description">
              FFA, PPA, and EBA predictions on subject-native cortex.
            </p>
          </div>
          <button type="button" className="brain-map-close" autoFocus onClick={onClose}>Close</button>
        </header>

        <div className="brain-map-stimulus-bar" aria-label="Brain-map stimulus">
          <span className="brain-map-stimulus-label">Focused stimulus</span>
          <div className="brain-map-stimulus-options">
            {session.stimuli.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === stimulus?.id ? 'active' : ''}
                aria-pressed={item.id === stimulus?.id}
                onClick={() => onFocusStimulus(item.id)}
              >
                <img src={item.imageUrl} alt="" />
                <span>{item.displayName}</span>
              </button>
            ))}
          </div>
          <p>One fixed scale across uploads and regions.</p>
        </div>

        <div className="brain-map-layout">
          <aside className="brain-map-controls" aria-label="Brain map controls">
            <fieldset>
              <legend>Cortical surface</legend>
              {(['inflated', 'pial'] as const).map((kind) => {
                const available = assetState.status !== 'ready' || Boolean(assetState.assets.manifest.surfaces[kind]);
                return (
                  <label key={kind}>
                    <input
                      type="radio"
                      name="brain-surface"
                      value={kind}
                      checked={surface === kind}
                      disabled={!available}
                      onChange={() => setSurface(kind)}
                    />
                    <span>{kind === 'inflated' ? 'Inflated' : 'Pial'}</span>
                  </label>
                );
              })}
              <small>Inflated exposes sulcal cortex; pial preserves native folding.</small>
            </fieldset>

            <fieldset>
              <legend>Brain regions</legend>
              {CANONICAL_ROIS.map((roi) => {
                const manifestRoi = assetState.status === 'ready' ? assetState.assets.manifest.rois[roi] : null;
                return (
                  <label key={roi}>
                    <input
                      type="checkbox"
                      checked={enabledRois[roi]}
                      onChange={(event) => setEnabledRois((current) => ({
                        ...current,
                        [roi]: event.target.checked,
                      }))}
                    />
                    <span><strong>{roi.toUpperCase()}</strong> — {ROI_NAMES[roi]}</span>
                    <small>{manifestRoi?.mappedResponseCount ?? '—'}/{manifestRoi?.responseCount ?? session.brainMapResults?.roiVoxelCounts[roi]}</small>
                  </label>
                );
              })}
            </fieldset>

            <div className="brain-map-scale" aria-label={`Fixed session predicted response scale from minus ${formatResponse(maxAbsResponse)} to plus ${formatResponse(maxAbsResponse)}`}>
              <div className="brain-map-scale-heading">
                <strong>Predicted response</strong>
                <span>model units · fixed scale</span>
              </div>
              <div className="brain-map-scale-gradient" aria-hidden="true" />
              <div className="brain-map-scale-labels">
                <span>−{formatResponse(maxAbsResponse)}</span>
                <span>0</span>
                <span>+{formatResponse(maxAbsResponse)}</span>
              </div>
            </div>

            <button type="button" className="brain-map-reset" onClick={() => setResetKey((key) => key + 1)}>
              Reset 3D view
            </button>
            <p className="brain-map-coverage">
              {mappedCount === null
                ? 'Checking surface coverage…'
                : `${mappedCount.toLocaleString()} of ${responseCount.toLocaleString()} voxel responses mapped. `}
              Three EBA responses are unmapped and not plotted.
            </p>
          </aside>

          <div className="brain-map-stage-column">
            <div className="brain-map-stage">
              {assetState.status === 'loading' && (
                <div className="brain-map-asset-state" role="status">
                  <span className="brain-map-spinner" aria-hidden="true" />
                  <strong>Loading Subject 1 anatomy…</strong>
                </div>
              )}
              {assetState.status === 'error' && (
                <div className="brain-map-asset-state error" role="alert">
                  <strong>Brain map unavailable</strong>
                  <p>{assetState.message}</p>
                  <p>No anatomical surface is shown when its metadata does not match the session response space.</p>
                </div>
              )}
              {assetState.status === 'ready' && stimulusResult && (
                <Suspense fallback={<div className="brain-map-asset-state" role="status">Preparing the 3D renderer…</div>}>
                  <BrainMapViewer
                    assets={assetState.assets}
                    stimulusResult={stimulusResult}
                    enabledRois={enabledRoiSet}
                    surface={surface}
                    maxAbsResponse={maxAbsResponse}
                    resetKey={resetKey}
                    onHover={setHovered}
                  />
                </Suspense>
              )}
              {hovered && (
                <div
                  className="brain-map-tooltip"
                  style={{ left: hovered.canvasX, top: hovered.canvasY } as CSSProperties}
                  role="status"
                >
                  <strong>{hovered.dominantSource.roi.toUpperCase()} · {hovered.hemisphere.toUpperCase()}</strong>
                  <span>Surface value: {formatResponse(hovered.surfaceResponse)}</span>
                  <span>Dominant source response: {formatResponse(hovered.dominantSource.response)}</span>
                  <span>ROI response index: {hovered.dominantSource.responseIndex}</span>
                  <span>Volume voxel [z, y, x]: [{hovered.volumeCoordinateZyx.join(', ')}]</span>
                  <span>Surface vertex: {hovered.vertexIndex.toLocaleString()}</span>
                  {hovered.overlappingRois.length > 1 && (
                    <span>ROI overlap: {hovered.overlappingRois.map((roi) => roi.toUpperCase()).join(', ')}</span>
                  )}
                </div>
              )}
            </div>
            <div className="brain-map-stage-footer">
              <span>Drag to rotate · scroll to zoom · hover a mapped vertex</span>
              <span>Keyboard: arrow keys rotate · +/− zoom</span>
            </div>
          </div>
        </div>

        <footer className="brain-map-science-note">
          <p>Model predictions, not measured BOLD signals for the uploaded image.</p>
          <details className="brain-map-mapping-details">
            <summary>Mapping details</summary>
            <p>
              Subject 1 fLoc-defined FFA, PPA, and EBA output voxels at <em>t</em> &gt; 7 are projected onto
              the subject-native cortical surface using a line-nearest-neighbor mapper.
            </p>
            <p>
              The NSD-1000 predictor was trained with pooled NSD subjects; this projection only locates
              its Subject 1 output columns. Color between source voxels is not anatomical activation.
              Three EBA responses lack mapper coverage and are omitted.
            </p>
          </details>
        </footer>
      </section>
    </div>,
    document.body,
  );
}

export function BrainMapDialog({ session, focusedStimulusId, onFocusStimulus }: BrainMapDialogProps) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  if (!session.brainMapResults) return null;

  function close() {
    setIsOpen(false);
    window.requestAnimationFrame(() => triggerRef.current?.focus());
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className="brain-map-trigger"
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        onClick={() => setIsOpen(true)}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M8.2 4.4a3.8 3.8 0 0 0-3.5 5.3 3.6 3.6 0 0 0 .5 6.8 3.5 3.5 0 0 0 5.8 2.2V5.8a3.8 3.8 0 0 0-2.8-1.4Zm7.6 0a3.8 3.8 0 0 1 3.5 5.3 3.6 3.6 0 0 1-.5 6.8 3.5 3.5 0 0 1-5.8 2.2V5.8a3.8 3.8 0 0 1 2.8-1.4Z" />
        </svg>
        3D Brain Map
      </button>
      {isOpen && (
        <BrainMapModal
          session={session}
          focusedStimulusId={focusedStimulusId}
          onFocusStimulus={onFocusStimulus}
          onClose={close}
        />
      )}
    </>
  );
}
