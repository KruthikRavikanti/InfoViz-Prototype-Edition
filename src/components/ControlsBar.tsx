import type { RankingSystem, Roi } from '../types/data';

type ControlsBarProps = {
  rois: string[];
  rankingSystem: RankingSystem;
  selectedRankingRoi: Roi;
  modelSearch: string;
  showScoreLabels: boolean;
  compareMode: boolean;
  compareSelectionCount: number;
  onRankingSystemChange: (rankingSystem: RankingSystem) => void;
  onSelectedRankingRoiChange: (roi: Roi) => void;
  onModelSearchChange: (searchTerm: string) => void;
  onShowScoreLabelsChange: (showScoreLabels: boolean) => void;
  onCompareModeChange: (enabled: boolean) => void;
  onResetControls: () => void;
};

export function ControlsBar({
  rois,
  rankingSystem,
  selectedRankingRoi,
  modelSearch,
  showScoreLabels,
  compareMode,
  compareSelectionCount,
  onRankingSystemChange,
  onSelectedRankingRoiChange,
  onModelSearchChange,
  onShowScoreLabelsChange,
  onCompareModeChange,
  onResetControls,
}: ControlsBarProps) {
  return (
    <section className="controls-bar" aria-label="Visualization controls">
      <label className="model-search-control">
        Search models
        <input
          type="search"
          value={modelSearch}
          placeholder="clip, resnet, vit…"
          onChange={(event) => onModelSearchChange(event.target.value)}
        />
      </label>

      <div className="model-ranking-controls">
        <label>
          Rank models
          <select value={rankingSystem} onChange={(event) => onRankingSystemChange(event.target.value as RankingSystem)}>
            <option value="overall">Overall alignment</option>
            <option value="roi">Selected ROI</option>
          </select>
        </label>

        {rankingSystem === 'roi' && (
          <label>
            ROI
            <select value={selectedRankingRoi} onChange={(event) => onSelectedRankingRoiChange(event.target.value)}>
              {rois.map((roi) => (
                <option key={roi} value={roi}>
                  {roi.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      <div className="model-display-controls">
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={showScoreLabels}
            onChange={(event) => onShowScoreLabelsChange(event.target.checked)}
          />
          Show scores
        </label>

        <label className="toggle-control">
          <input
            type="checkbox"
            checked={compareMode}
            onChange={(event) => onCompareModeChange(event.target.checked)}
          />
          Compare selections
        </label>
      </div>

      {compareMode && (
        <div className="compare-status active" aria-live="polite">
          {compareSelectionCount} / 2 selected
        </div>
      )}

      <button className="reset-controls-button" type="button" onClick={onResetControls}>
        Reset
      </button>
    </section>
  );
}
