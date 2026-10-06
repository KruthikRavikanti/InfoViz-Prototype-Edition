import { CompareDrawer } from './CompareDrawer';
import { ControlsBar } from './ControlsBar';
import { DetailsPanel } from './DetailsPanel';
import { Heatmap } from './Heatmap';
import { HelpBox } from './HelpBox';
import { SessionBanner } from './SessionBanner';
import type {
  AggregateHeatmapCell,
  ClusteringData,
  ModelRoiColumn,
  RankingSystem,
  Roi,
  SelectedHeatmapCell,
  SortDirection,
  WideCsvRow,
} from '../types/data';
import type { CortexComparisonSession } from '../types/session';

type VisualizationViewProps = {
  rows: WideCsvRow[];
  clustering: ClusteringData;
  modelRoiColumns: ModelRoiColumn[];
  heatmapRois: Roi[];
  heatmapCells: AggregateHeatmapCell[];
  heatmapColorDomain: readonly [number, number];
  visibleModels: string[];
  rankingSystem: RankingSystem;
  selectedRankingRoi: Roi;
  sortDirection: SortDirection;
  modelSearch: string;
  showScoreLabels: boolean;
  selectedCell: SelectedHeatmapCell | null;
  compareMode: boolean;
  compareCells: SelectedHeatmapCell[];
  comparableCellIds: Set<string>;
  session: CortexComparisonSession | null;
  focusedStimulusId: string | null;
  sessionCellIds: Set<string>;
  onRankingSystemChange: (rankingSystem: RankingSystem) => void;
  onSelectedRankingRoiChange: (roi: Roi) => void;
  onSortDirectionToggle: () => void;
  onModelSearchChange: (searchTerm: string) => void;
  onShowScoreLabelsChange: (showScoreLabels: boolean) => void;
  onResetControls: () => void;
  onSelectCell: (cell: SelectedHeatmapCell) => void;
  onCompareModeChange: (enabled: boolean) => void;
  onFocusStimulus: (stimulusId: string) => void;
};

export function VisualizationView({
  rows,
  clustering,
  modelRoiColumns,
  heatmapRois,
  heatmapCells,
  heatmapColorDomain,
  visibleModels,
  rankingSystem,
  selectedRankingRoi,
  sortDirection,
  modelSearch,
  showScoreLabels,
  selectedCell,
  compareMode,
  compareCells,
  comparableCellIds,
  session,
  focusedStimulusId,
  sessionCellIds,
  onRankingSystemChange,
  onSelectedRankingRoiChange,
  onSortDirectionToggle,
  onModelSearchChange,
  onShowScoreLabelsChange,
  onResetControls,
  onSelectCell,
  onCompareModeChange,
  onFocusStimulus,
}: VisualizationViewProps) {
  return (
    <>
      {session && (
        <SessionBanner
          session={session}
          focusedStimulusId={focusedStimulusId}
          onFocusStimulus={onFocusStimulus}
        />
      )}
      <div className="workspace-grid">
        <aside className="model-browser" aria-label="Model browser">
          <ControlsBar
            rois={heatmapRois}
            rankingSystem={rankingSystem}
            selectedRankingRoi={selectedRankingRoi}
            modelSearch={modelSearch}
            showScoreLabels={showScoreLabels}
            compareMode={compareMode}
            compareSelectionCount={compareCells.length}
            onRankingSystemChange={onRankingSystemChange}
            onSelectedRankingRoiChange={onSelectedRankingRoiChange}
            onModelSearchChange={onModelSearchChange}
            onShowScoreLabelsChange={onShowScoreLabelsChange}
            onCompareModeChange={onCompareModeChange}
            onResetControls={onResetControls}
          />
          <Heatmap
            cells={heatmapCells}
            colorDomain={heatmapColorDomain}
            models={visibleModels}
            rois={heatmapRois}
            showScoreLabels={showScoreLabels}
            sortDirection={sortDirection}
            selectedCell={selectedCell}
            compareMode={compareMode}
            compareCells={compareCells}
            comparableCellIds={comparableCellIds}
            sessionCellIds={sessionCellIds}
            onSortDirectionToggle={onSortDirectionToggle}
            onSelectCell={onSelectCell}
            selectedRankingRoi={rankingSystem === 'roi' ? selectedRankingRoi : null}
          />
          <HelpBox />
        </aside>
        <DetailsPanel
          heatmapCells={heatmapCells}
          imageCount={rows.length}
          clustering={clustering}
          modelRoiColumns={modelRoiColumns}
          onSelectCell={onSelectCell}
          rows={rows}
          selectedCell={selectedCell}
          session={session}
          focusedStimulusId={focusedStimulusId}
          onFocusStimulus={onFocusStimulus}
        />
      </div>
      <CompareDrawer
        heatmapCells={heatmapCells}
        compareMode={compareMode}
        compareCells={compareCells}
        clustering={clustering}
        rows={rows}
        modelRoiColumns={modelRoiColumns}
        topK={6}
        onCompareModeChange={onCompareModeChange}
      />
    </>
  );
}
