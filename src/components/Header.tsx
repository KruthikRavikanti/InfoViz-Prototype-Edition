type HeaderProps = {
  imageCount: number;
  modelCount: number;
  roiCount: number;
};

export function Header({ imageCount, modelCount, roiCount }: HeaderProps) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">Murty Lab · Vision AI</p>
        <h1>Model Performance Dashboard</h1>
        <p className="lede">
          Aggregate alignment scores across cortical regions — click any cell to inspect image-level evidence.
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
