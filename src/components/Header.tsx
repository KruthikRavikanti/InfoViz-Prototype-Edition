import { publicPath } from '../utils/publicPath';
import type { SelectedHeatmapCell } from '../types/data';
import type { CortexComparisonSession } from '../types/session';
import { isStochasticReferenceModel, STOCHASTIC_REFERENCE_NOTE } from '../utils/predictionPolicy';

type HeaderProps = {
  imageCount: number;
  modelCount: number;
  roiCount: number;
  selectedCell?: SelectedHeatmapCell | null;
  hasImageResponses?: boolean;
  compareMode?: boolean;
  compareCells?: SelectedHeatmapCell[];
  session?: CortexComparisonSession | null;
};

function mappingLabel(dataset: string): string {
  if (dataset === 'nsd_1000') return 'NSD-1000 mapping';
  if (dataset === 'murty185') return 'Murty-185 mapping';
  return dataset;
}

export function Header({
  imageCount,
  modelCount,
  roiCount,
  selectedCell = null,
  hasImageResponses = false,
  compareMode = false,
  compareCells = [],
  session = null,
}: HeaderProps) {
  const cortexHome = `${(import.meta.env.VITE_CORTEX_APP_URL?.trim() || '/').replace(/\/+$/, '')}/`;
  const cortexLab = `${cortexHome}lab/`;
  const viewingUploadModel = Boolean(session && selectedCell
    && selectedCell.model === session.comparison.model
    && session.comparison.rois.some((roi) => roi === selectedCell.roi));
  const showUploadModelSeparately = session && (compareMode || (selectedCell && !viewingUploadModel));
  const displayedModels = compareMode
    ? compareCells.map((cell) => cell.model)
    : selectedCell && hasImageResponses ? [selectedCell.model] : session ? [session.comparison.model] : [];
  const showStochasticNote = displayedModels.some(isStochasticReferenceModel);

  return (
    <header className="app-header">
      <div className="app-masthead">
        <a className="cortex-brand" href={cortexHome} aria-label="Virtual Visual Cortex home">
          <img src={publicPath('cortex-logo.svg')} alt="" />
          <span>Cortex</span>
        </a>
        <nav className="app-masthead-nav" aria-label="Cortex navigation">
          <span className="app-product-label">Cortex-Compass</span>
          <a className="back-to-lab" href={cortexLab}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M19 12H5m7-7-7 7 7 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Back to the Lab
          </a>
        </nav>
      </div>
      <div className="app-header-content">
        <div>
          <h1>Cortex-Compass</h1>
          <div className="header-context" aria-label="Reference and model context">
            <p className="lede">
              <span><strong>Reference:</strong> Murty-185 ({imageCount} images)</span>
              <span aria-hidden="true"> · </span>
              {compareMode ? (
                <span>
                  {compareCells.length > 0 ? (
                    <><strong>Reference comparison:</strong> {compareCells.map((cell) => `${cell.model} / ${cell.roi.toUpperCase()}`).join(' ↔ ')} · NSD-1000 predictions · reference images only{compareCells.length === 1 ? ' · choose a second cell' : ''}</>
                  ) : 'Choose two model–region cells to compare (reference images only).'}
                </span>
              ) : selectedCell ? (
                <span>
                  <strong>Viewing:</strong> {selectedCell.model} · {selectedCell.roi.toUpperCase()} ·{' '}
                  {hasImageResponses ? 'NSD-1000 predictions' : 'alignment scores only'}
                  {session && hasImageResponses && !viewingUploadModel ? ' · reference images only' : ''}
                </span>
              ) : session ? (
                <span>
                  <strong>Upload comparison:</strong> {session.comparison.model} · {mappingLabel(session.comparison.dataset)} ·{' '}
                  {session.comparison.rois.map((roi) => roi.toUpperCase()).join(', ')}
                </span>
              ) : <span>Select a model and region to explore predictions.</span>}
            </p>
            {session && (
              <p className="lede header-lab-settings">
                <strong>Your Lab settings:</strong> {session.sourceRequest.model} · {mappingLabel(session.sourceRequest.dataset)} · {session.sourceRequest.roi.toUpperCase()}
                {showUploadModelSeparately && (
                  <> · <strong>Uploads:</strong> {session.comparison.model} · {mappingLabel(session.comparison.dataset)}</>
                )}
              </p>
            )}
            {showStochasticNote && (
              <p className="lede header-lab-settings">
                {STOCHASTIC_REFERENCE_NOTE}
              </p>
            )}
          </div>
        </div>
        <dl className="summary-stats" aria-label="Dataset summary">
          <div>
            <dt>References</dt>
            <dd>{imageCount || '—'}</dd>
          </div>
          <div>
            <dt>Models</dt>
            <dd>{modelCount || '—'}</dd>
          </div>
          <div>
            <dt>ROIs</dt>
            <dd>{roiCount || '—'}</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}
