import { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { TabNav } from './components/TabNav';
import type { AppTab } from './components/TabNav';
import { UserStudyView } from './components/UserStudyView';
import { VisualizationView } from './components/VisualizationView';
import type { RankingSystem, Roi, SelectedHeatmapCell, SortDirection, VisualizationData } from './types/data';
import { loadVisualizationData } from './utils/data';
import { hasComparableData } from './utils/evidence';
import {
  buildAggregateHeatmapCells,
  filterModelsBySearch,
  getDefaultHeatmapRois,
  inferAggregateModels,
  orderAggregateModels,
} from './utils/heatmap';

type DataState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: VisualizationData };

export default function App() {
  const [dataState, setDataState] = useState<DataState>({ status: 'loading' });
  const [rankingSystem, setRankingSystem] = useState<RankingSystem>('overall');
  const [selectedRankingRoi, setSelectedRankingRoi] = useState<Roi>('ffa');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [modelSearch, setModelSearch] = useState('');
  const [showScoreLabels, setShowScoreLabels] = useState(false);
  const [selectedCell, setSelectedCell] = useState<SelectedHeatmapCell | null>(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareCells, setCompareCells] = useState<SelectedHeatmapCell[]>([]);
  const [activeTab, setActiveTab] = useState<AppTab>('visualization');

  useEffect(() => {
    let ignore = false;

    loadVisualizationData()
      .then((data) => {
        if (!ignore) {
          setDataState({ status: 'ready', data });
        }
      })
      .catch((error: unknown) => {
        if (!ignore) {
          setDataState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Unable to load visualization data.',
          });
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  function handleCompareModeChange(enabled: boolean) {
    setCompareMode(enabled);
    setCompareCells([]);

    if (enabled) {
      setSelectedCell(null);
    }
  }

  function handleHeatmapCellClick(cell: SelectedHeatmapCell) {
    if (!compareMode) {
      setSelectedCell(cell);
      return;
    }

    setCompareCells((currentCells) => {
      if (currentCells.some((compareCell) => compareCell.id === cell.id)) {
        return currentCells.filter((compareCell) => compareCell.id !== cell.id);
      }

      if (currentCells.length < 2) {
        return [...currentCells, cell];
      }

      return [currentCells[1], cell];
    });
  }

  function handleResetControls() {
    setRankingSystem('overall');
    setSelectedRankingRoi('ffa');
    setSortDirection('desc');
    setModelSearch('');
    setShowScoreLabels(false);
    setSelectedCell(null);
    setCompareMode(false);
    setCompareCells([]);
  }

  if (dataState.status === 'loading') {
    return (
      <main className="app-shell">
        <div className="status-panel">
          <p className="eyebrow">Loading</p>
          <h1>Reading model response data</h1>
          <p>Loading CSV rows, aggregate ROI scores, and clustering outputs from `public/data`.</p>
        </div>
      </main>
    );
  }

  if (dataState.status === 'error') {
    return (
      <main className="app-shell">
        <div className="status-panel error-panel">
          <p className="eyebrow">Data error</p>
          <h1>Unable to load visualization data</h1>
          <p>{dataState.message}</p>
        </div>
      </main>
    );
  }

  const { data } = dataState;
  const heatmapRois = getDefaultHeatmapRois(data.scores);
  const aggregateModels = inferAggregateModels(data.scores, heatmapRois);
  const rankingRoi: Roi = heatmapRois.includes(selectedRankingRoi) ? selectedRankingRoi : (heatmapRois[0] ?? 'ffa');
  const orderedModels = orderAggregateModels(aggregateModels, data.scores, rankingSystem, rankingRoi, sortDirection);
  const visibleModels = filterModelsBySearch(orderedModels, modelSearch);
  const heatmapCells = buildAggregateHeatmapCells(data.scores, heatmapRois, visibleModels);
  const comparableCellIds = new Set(
    heatmapCells
      .filter((cell) => hasComparableData(cell.model, cell.roi, data.rows, data.modelRoiColumns))
      .map((cell) => cell.id),
  );

  return (
    <main className="app-shell">
      <Header imageCount={data.rows.length} modelCount={aggregateModels.length} roiCount={heatmapRois.length} />
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />
      <section className="tab-panel" hidden={activeTab !== 'visualization'} aria-label="Visualization view">
        <VisualizationView
          rows={data.rows}
          clustering={data.clustering}
          modelRoiColumns={data.modelRoiColumns}
          heatmapRois={heatmapRois}
          heatmapCells={heatmapCells}
          visibleModels={visibleModels}
          rankingSystem={rankingSystem}
          selectedRankingRoi={rankingRoi}
          sortDirection={sortDirection}
          modelSearch={modelSearch}
          showScoreLabels={showScoreLabels}
          selectedCell={selectedCell}
          compareMode={compareMode}
          compareCells={compareCells}
          comparableCellIds={comparableCellIds}
          onRankingSystemChange={setRankingSystem}
          onSelectedRankingRoiChange={setSelectedRankingRoi}
          onSortDirectionToggle={() => setSortDirection((currentDirection) => (currentDirection === 'desc' ? 'asc' : 'desc'))}
          onModelSearchChange={setModelSearch}
          onShowScoreLabelsChange={setShowScoreLabels}
          onResetControls={handleResetControls}
          onSelectCell={handleHeatmapCellClick}
          onCompareModeChange={handleCompareModeChange}
        />
      </section>
      <section className="tab-panel" hidden={activeTab !== 'user-study'} aria-label="User study view">
        <UserStudyView />
      </section>
    </main>
  );
}
