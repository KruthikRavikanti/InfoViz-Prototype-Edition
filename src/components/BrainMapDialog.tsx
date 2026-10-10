import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { BrainMapTooltip } from './BrainMapTooltip';
import { loadBrainMapAssets } from '../services/brainMapAssets';
import { BRAIN_REFERENCE_VERSION, loadBrainMapReferences } from '../services/brainReferenceAssets';
import { isStochasticReferenceModel, STOCHASTIC_REFERENCE_NOTE } from '../utils/predictionPolicy';
import type { BrainMapAssets, BrainMapHoverDatum, BrainMapReferenceSet, BrainSurfaceKind } from '../types/brainMap';
import {
  CANONICAL_ROIS,
  SUBJECT_ONE_ROI_VOXEL_COUNTS,
  type CanonicalRoi,
  type CortexComparisonSession,
} from '../types/session';
import { fixedSessionResponseExtent } from '../utils/brainProjection';
import { buildBrainVoxelImagePeaks } from '../utils/brainVoxelImagePeaks';
import {
  brainMapImageNeighbors,
  brainMapReferenceImages,
  brainMapUploadedImages,
  filterBrainMapImages,
  type BrainMapImage,
  type BrainMapImageSource,
} from '../utils/brainMapSelection';

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
  session?: CortexComparisonSession | null;
  selectedModel?: string;
  focusedStimulusId?: string | null;
  onFocusStimulus?: (stimulusId: string) => void;
};

type ReferenceState =
  | { model: string; status: 'loading' }
  | { model: string; status: 'error'; message: string }
  | { model: string; status: 'ready'; references: BrainMapReferenceSet };

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
  selectedModel,
  focusedStimulusId,
  onFocusStimulus,
  onClose,
}: BrainMapDialogProps & { onClose: () => void }) {
  const model = selectedModel ?? session?.comparison.model ?? '';
  const [assetState, setAssetState] = useState<AssetState>({ status: 'loading' });
  const [referenceState, setReferenceState] = useState<ReferenceState>({ model, status: 'loading' });
  const [surface, setSurface] = useState<BrainSurfaceKind>('inflated');
  const [enabledRois, setEnabledRois] = useState<Record<CanonicalRoi, boolean>>({
    ffa: true,
    ppa: true,
    eba: true,
  });
  const [hovered, setHovered] = useState<BrainMapHoverDatum | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [imageSource, setImageSource] = useState<BrainMapImageSource>(() => (
    brainMapUploadedImages(session, model).length ? 'uploaded' : 'reference'
  ));
  const [referenceId, setReferenceId] = useState<string>();
  const [localUploadId, setLocalUploadId] = useState<string>();
  const [imageSearch, setImageSearch] = useState('');
  const [controlsOpen, setControlsOpen] = useState(() => (
    typeof window === 'undefined' || typeof window.matchMedia !== 'function'
      ? true : window.matchMedia('(min-width: 901px)').matches
  ));
  const references = referenceState.model === model && referenceState.status === 'ready'
    ? referenceState.references : undefined;
  const uploadedImages = useMemo(
    () => brainMapUploadedImages(session, model, references),
    [session, model, references],
  );
  const referenceImages = useMemo(() => brainMapReferenceImages(references), [references]);
  const activeSource = imageSource === 'uploaded' && uploadedImages.length ? 'uploaded' : 'reference';
  const images = activeSource === 'uploaded' ? uploadedImages : referenceImages;
  // Search only narrows the picker, not the collection used for voxel rankings.
  // Precompute once per collection so pointer movement only performs a lookup.
  const imagePeaks = useMemo(() => buildBrainVoxelImagePeaks(images, activeSource), [images, activeSource]);
  const selectedId = activeSource === 'uploaded' ? focusedStimulusId ?? localUploadId : referenceId;
  const stimulus = images.find((item) => item.id === selectedId) ?? images[0];
  const stimulusResult = stimulus?.result;
  const filteredImages = useMemo(() => filterBrainMapImages(images, imageSearch), [images, imageSearch]);
  const imageNeighbors = brainMapImageNeighbors(filteredImages, stimulus?.id);
  const selectedFilteredId = filteredImages.some((image) => image.id === stimulus?.id) ? stimulus?.id : '';
  const enabledRoiSet = useMemo(
    () => new Set(CANONICAL_ROIS.filter((roi) => enabledRois[roi])),
    [enabledRois],
  );
  const maxAbsResponse = useMemo(
    () => fixedSessionResponseExtent([...uploadedImages, ...referenceImages].map((image) => image.result)),
    [uploadedImages, referenceImages],
  );

  function selectImage(image: BrainMapImage) {
    setHovered(null);
    if (image.source === 'uploaded') {
      setLocalUploadId(image.id);
      onFocusStimulus?.(image.id);
    } else setReferenceId(image.id);
  }

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
    let ignore = false;
    const abortController = new AbortController();
    setReferenceState({ model, status: 'loading' });
    loadBrainMapReferences(model, fetch, abortController.signal)
      .then((loaded) => {
        if (!ignore) setReferenceState({ model, status: 'ready', references: loaded });
      })
      .catch((error: unknown) => {
        if (ignore) return;
        setReferenceState({
          model,
          status: 'error',
          message: error instanceof Error ? error.message : 'Reference predictions could not be loaded.',
        });
      });
    return () => {
      ignore = true;
      abortController.abort();
    };
  }, [model]);

  useEffect(() => {
    setHovered(null);
  }, [stimulusResult]);

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia('(min-width: 901px)');
    const updateControls = () => setControlsOpen(media.matches);
    media.addEventListener('change', updateControls);
    return () => media.removeEventListener('change', updateControls);
  }, []);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previousOverflow; };
  }, []);

  const mappedCount = assetState.status === 'ready'
    ? CANONICAL_ROIS.reduce((sum, roi) => sum + assetState.assets.manifest.rois[roi].mappedResponseCount, 0)
    : null;
  const responseCount = CANONICAL_ROIS.reduce((sum, roi) => sum + SUBJECT_ONE_ROI_VOXEL_COUNTS[roi], 0);
  const referenceIsLoading = referenceState.model !== model || referenceState.status === 'loading';
  const referenceError = referenceState.model === model && referenceState.status === 'error'
    ? referenceState.message : null;

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
              {model} · FFA, PPA, and EBA predictions on subject-native cortex.
            </p>
            {isStochasticReferenceModel(model) ? <p>{STOCHASTIC_REFERENCE_NOTE}</p> : null}
          </div>
          <button type="button" className="brain-map-close" autoFocus onClick={onClose}>Close</button>
        </header>

        <div className="brain-map-stimulus-bar" aria-label="Brain-map image selection">
          <div className="brain-map-picker">
            <div className="brain-map-picker-row">
              <label className="brain-map-image-source">
                <span>Image collection</span>
                <select
                  value={activeSource}
                  onChange={(event) => {
                    setImageSource(event.target.value as BrainMapImageSource);
                    setImageSearch('');
                    setHovered(null);
                  }}
                >
                  <option value="uploaded" disabled={!uploadedImages.length}>Uploads ({uploadedImages.length})</option>
                  <option value="reference">Murty-185 reference ({references ? referenceImages.length : '185'})</option>
                </select>
              </label>
              <label className="brain-map-search">
                <span className="sr-only">Search brain-map images</span>
                <input
                  type="search"
                  placeholder="Search images…"
                  value={imageSearch}
                  onChange={(event) => setImageSearch(event.target.value)}
                />
              </label>
            </div>
            <div className="brain-map-picker-row">
              <label className="brain-map-image-select">
                <span className="sr-only">Brain-map image</span>
                <select
                  value={selectedFilteredId ?? ''}
                  disabled={!filteredImages.length}
                  onChange={(event) => {
                    const image = filteredImages.find((item) => item.id === event.target.value);
                    if (image) selectImage(image);
                  }}
                >
                  <option value="" disabled>
                    {referenceIsLoading && activeSource === 'reference'
                      ? 'Loading reference predictions…'
                      : filteredImages.length ? 'Select an image…' : 'No matching images'}
                  </option>
                  {filteredImages.map((image) => (
                    <option key={image.id} value={image.id}>{image.displayName}</option>
                  ))}
                </select>
              </label>
              <div className="brain-map-image-navigation">
                <button
                  type="button"
                  aria-label="Previous brain-map image"
                  disabled={!imageNeighbors.previous}
                  onClick={() => { if (imageNeighbors.previous) selectImage(imageNeighbors.previous); }}
                >←</button>
                <button
                  type="button"
                  aria-label="Next brain-map image"
                  disabled={!imageNeighbors.next}
                  onClick={() => { if (imageNeighbors.next) selectImage(imageNeighbors.next); }}
                >→</button>
              </div>
              <span className="brain-map-image-count" role="status">
                {imageSearch.trim() ? `${filteredImages.length} found` : `${images.length} images`}
              </span>
            </div>
            {referenceError && (
              <p className="brain-map-reference-error" role="status">Reference predictions unavailable: {referenceError}</p>
            )}
            {session?.brainMapResults && session.comparison.model !== model && (
              <p className="brain-map-reference-error">Select {session.comparison.model} to view this session’s uploads.</p>
            )}
            {session?.brainMapResults && session.comparison.model === model
              && BRAIN_REFERENCE_VERSION !== session.comparison.referenceVersion && (
              <p className="brain-map-reference-error">Uploads use a different reference bundle and cannot share this brain map.</p>
            )}
          </div>
          <div className="brain-map-current-image" aria-live="polite">
            {stimulus && <img src={stimulus.imageUrl} alt={stimulus.displayName} />}
            <div>
              <span className="brain-map-context-label">Shown on brain</span>
              <strong>{stimulus?.displayName ?? 'Choose an image'}</strong>
              <span>{activeSource === 'uploaded' ? 'Uploaded stimulus' : 'Murty-185 reference'}</span>
            </div>
          </div>
        </div>

        <div className="brain-map-layout">
          <aside className="brain-map-controls" aria-label="Brain map controls">
            <details
              className="brain-map-control-disclosure"
              open={controlsOpen}
              onToggle={(event) => setControlsOpen(event.currentTarget.open)}
            >
              <summary>Display controls</summary>
              <div className="brain-map-control-fields">
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
                    <small>{manifestRoi?.mappedResponseCount ?? '—'}/{manifestRoi?.responseCount ?? '—'}</small>
                  </label>
                );
              })}
            </fieldset>

            <div className="brain-map-scale" aria-busy={referenceIsLoading} aria-label={referenceIsLoading
              ? 'Preparing shared predicted response scale'
              : `Fixed image collection predicted response scale from minus ${formatResponse(maxAbsResponse)} to plus ${formatResponse(maxAbsResponse)}`}>
              <div className="brain-map-scale-heading">
                <strong>Selected image response</strong>
                <span>{referenceIsLoading ? 'preparing scale' : 'model units · fixed scale'}</span>
              </div>
              <div className="brain-map-scale-gradient" aria-hidden="true" />
              <div className="brain-map-scale-labels">
                {referenceIsLoading ? <span>Loading image collection…</span> : <>
                  <span>−{formatResponse(maxAbsResponse)}</span>
                  <span>0</span>
                  <span>+{formatResponse(maxAbsResponse)}</span>
                </>}
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
              </div>
            </details>
          </aside>

          <div className="brain-map-stage-column">
            <div className="brain-map-stage" data-image-id={stimulus?.id} data-image-source={activeSource} aria-busy={assetState.status === 'loading' || referenceIsLoading}>
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
              {assetState.status === 'ready' && stimulusResult && !referenceIsLoading && (
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
              {assetState.status === 'ready' && (referenceIsLoading || !stimulusResult) && (
                <div className={`brain-map-asset-state${referenceError ? ' error' : ''}`} role={referenceError ? 'alert' : 'status'}>
                  <strong>{referenceError ? 'Reference brain map unavailable' : 'Preparing shared response scale…'}</strong>
                  <p>{referenceError ?? 'Preparing the verified Subject 1 voxel responses for all 185 reference images.'}</p>
                </div>
              )}
              {hovered && stimulus && (
                <BrainMapTooltip
                  hovered={hovered}
                  imagePeaks={imagePeaks}
                  displayedImage={stimulus}
                  onDismiss={() => setHovered(null)}
                />
              )}
            </div>
            <div className="brain-map-stage-footer">
              <div>
                <strong>Hover previews do not change the brain.</strong>
                <span>Colors show the selected image · hover compares images at a voxel</span>
              </div>
              <div>
                <span>Drag to rotate · scroll to zoom</span>
                <span>Keyboard: arrow keys rotate · +/− zoom</span>
              </div>
            </div>
          </div>
        </div>

        <footer className="brain-map-science-note">
          <p>Model predictions, not measured BOLD signals for these images.</p>
          <details className="brain-map-mapping-details">
            <summary>Mapping details</summary>
            <p>
              Hover previews rank raw predicted responses for the displayed image’s dominant source voxel,
              not the blended surface vertex. Only the active image collection is compared; searching
              narrows the image picker, not this ranking. Hovering does not change the displayed brain image.
            </p>
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

export function BrainMapDialog({ session, selectedModel, focusedStimulusId, onFocusStimulus }: BrainMapDialogProps) {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  if (!(selectedModel ?? session?.comparison.model)) return null;

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
          selectedModel={selectedModel}
          focusedStimulusId={focusedStimulusId}
          onFocusStimulus={onFocusStimulus}
          onClose={close}
        />
      )}
    </>
  );
}
