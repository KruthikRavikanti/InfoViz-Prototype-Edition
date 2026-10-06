import { publicPath } from '../utils/publicPath';

type HeaderProps = {
  imageCount: number;
  modelCount: number;
  roiCount: number;
};

export function Header({ imageCount, modelCount, roiCount }: HeaderProps) {
  const cortexHome = `${(import.meta.env.VITE_CORTEX_APP_URL?.trim() || '/').replace(/\/+$/, '')}/`;
  const cortexLab = `${cortexHome}lab/`;

  return (
    <header className="app-header">
      <div className="app-masthead">
        <a className="cortex-brand" href={cortexHome} aria-label="Virtual Visual Cortex home">
          <img src={publicPath('cortex-logo.svg')} alt="" />
          <span>Cortex</span>
        </a>
        <nav className="app-masthead-nav" aria-label="Cortex navigation">
          <span className="app-product-label">Visualization</span>
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
          <p className="eyebrow">Murty Lab · Vision AI</p>
          <h1>Model &amp; Stimulus Explorer</h1>
          <p className="lede">
            Compare model alignment and explore image responses across cortical regions.
          </p>
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
