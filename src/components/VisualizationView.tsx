import { CompareDrawer } from './CompareDrawer';
import { ControlsBar } from './ControlsBar';
import { DetailsPanel } from './DetailsPanel';
import { Heatmap } from './Heatmap';
import { HelpBox } from './HelpBox';
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

type VisualizationViewProps = {
  rows: WideCsvRow[];
  clustering: ClusteringData;
  modelRoiColumns: ModelRoiColumn[];
  heatmapRois: Roi[];
  heatmapCells: AggregateHeatmapCell[];
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
  onRankingSystemChange: (rankingSystem: RankingSystem) => void;
  onSelectedRankingRoiChange: (roi: Roi) => void;
  onSortDirectionToggle: () => void;
  onModelSearchChange: (searchTerm: string) => void;
  onShowScoreLabelsChange: (showScoreLabels: boolean) => void;
  onResetControls: () => void;
  onSelectCell: (cell: SelectedHeatmapCell) => void;
  onCompareModeChange: (enabled: boolean) => void;
};

export function VisualizationView({
  rows,
  clustering,
  modelRoiColumns,
  heatmapRois,
  heatmapCells,
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
  onRankingSystemChange,
  onSelectedRankingRoiChange,
  onSortDirectionToggle,
  onModelSearchChange,
  onShowScoreLabelsChange,
  onResetControls,
  onSelectCell,
  onCompareModeChange,
}: VisualizationViewProps) {
  return (
    <>
      <HelpBox />
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
      <div className="workspace-grid">
        <Heatmap
          cells={heatmapCells}
          models={visibleModels}
          rois={heatmapRois}
          showScoreLabels={showScoreLabels}
          sortDirection={sortDirection}
          selectedCell={selectedCell}
          compareMode={compareMode}
          compareCells={compareCells}
          comparableCellIds={comparableCellIds}
          onSortDirectionToggle={onSortDirectionToggle}
          onSelectCell={onSelectCell}
          selectedRankingRoi={rankingSystem === 'roi' ? selectedRankingRoi : null}
        />
        <DetailsPanel
          heatmapCells={heatmapCells}
          imageCount={rows.length}
          clustering={clustering}
          modelRoiColumns={modelRoiColumns}
          onSelectCell={onSelectCell}
          rows={rows}
          selectedCell={selectedCell}
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
