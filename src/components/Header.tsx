type HeaderProps = {
  imageCount: number;
  modelCount: number;
  roiCount: number;
};

export function Header({ imageCount, modelCount, roiCount }: HeaderProps) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">Murty Lab &middot; Neuroscience &times; Vision AI</p>
        <div className="header-title-row">
          <h1>Vision-AI Performance Visualization</h1>
        </div>
        <p className="lede">
          Explore aggregate model alignment scores across cortical ROIs and inspect image-level evidence.
        </p>
      </div>
      <dl className="summary-stats" aria-label="Dataset summary">
        <div>
          <dt>Images</dt>
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
    </header>
  );
}
