import { useEffect, useRef, useState } from 'react';
import { Header } from './components/Header';
import { TabNav } from './components/TabNav';
import type { AppTab } from './components/TabNav';
import { UserStudyView } from './components/UserStudyView';
import { VisualizationView } from './components/VisualizationView';
import type { RankingSystem, Roi, SelectedHeatmapCell, SortDirection, VisualizationData } from './types/data';
import { loadVisualizationData } from './utils/data';
import { hasComparableData } from './utils/evidence';
import type { SessionLoadState } from './types/session';
import {
  CortexSessionError,
  assertSessionReferenceCompatible,
  clearSessionFragment,
  extractSessionToken,
  getCortexAppUrl,
  loadCortexComparisonSession,
  loadPublicReferenceManifest,
} from './services/cortexSession';
import { isCanonicalRoi } from './utils/sessionEvidence';
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

function initialSessionState(): SessionLoadState {
  try {
    const token = extractSessionToken(window.location.hash);
    return token ? { status: 'loading', token } : { status: 'static' };
  } catch (error) {
    return {
      status: 'error',
      kind: error instanceof CortexSessionError ? error.kind : 'invalid',
      message: error instanceof Error ? error.message : 'The visualization session token is invalid.',
    };
  }
}

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
  const [sessionState, setSessionState] = useState<SessionLoadState>(initialSessionState);
  const [focusedStimulusId, setFocusedStimulusId] = useState<string | null>(null);
  const initializedSessionVersion = useRef<string | null>(null);

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

  useEffect(() => {
    if (sessionState.status !== 'loading') return;
    let ignore = false;

    Promise.all([
      loadCortexComparisonSession(sessionState.token),
      loadPublicReferenceManifest(),
    ])
      .then(([session, manifest]) => {
        assertSessionReferenceCompatible(session, manifest);
        if (ignore) return;
        clearSessionFragment();
        setFocusedStimulusId(session.stimuli[0]?.id ?? null);
        setSessionState({ status: 'ready', session });
      })
      .catch((error: unknown) => {
        if (ignore) return;
        setSessionState({
          status: 'error',
          kind: error instanceof CortexSessionError ? error.kind : 'unavailable',
          message: error instanceof Error ? error.message : 'The Cortex comparison session could not be loaded.',
        });
      });

    return () => { ignore = true; };
  }, [sessionState.status === 'loading' ? sessionState.token : null]);

  useEffect(() => {
    if (dataState.status !== 'ready' || sessionState.status !== 'ready') return;
    const version = sessionState.session.comparison.referenceVersion;
    if (initializedSessionVersion.current === version) return;

    const { scores } = dataState.data;
    const rois = getDefaultHeatmapRois(scores);
    const models = inferAggregateModels(scores, rois);
    const sourceRoi = sessionState.session.sourceRequest.roi.toLowerCase();
    const initialRoi: Roi = isCanonicalRoi(sourceRoi)
      ? sourceRoi
      : 'ffa';
    const initialCell = buildAggregateHeatmapCells(scores, rois, models)
      .find((cell) => cell.model === sessionState.session.comparison.model && cell.roi === initialRoi);
    if (initialCell) setSelectedCell(initialCell);
    setSelectedRankingRoi(initialRoi);
    setActiveTab('visualization');
    initializedSessionVersion.current = version;
  }, [dataState, sessionState]);

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

  if (sessionState.status === 'error') {
    return (
      <main className="app-shell session-error-shell">
        <div className="status-panel error-panel session-error-panel">
          <p className="eyebrow">Cortex session unavailable</p>
          <h1>{sessionState.kind === 'expired' ? 'This comparison session expired' : 'Unable to open this comparison'}</h1>
          <p>{sessionState.message}</p>
          <a className="session-return-link" href={getCortexAppUrl()}>Return to Virtual Cortex</a>
        </div>
      </main>
    );
  }

  if (dataState.status === 'loading' || sessionState.status === 'loading') {
    return (
      <main className="app-shell">
        <div className="status-panel">
          <p className="eyebrow">Loading</p>
          <h1>{sessionState.status === 'loading' ? 'Loading your comparison' : 'Loading the explorer'}</h1>
          <p>{sessionState.status === 'loading' ? 'Preparing uploaded stimuli and Murty reference images.' : 'Preparing model scores and image responses.'}</p>
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
  const session = sessionState.status === 'ready' ? sessionState.session : null;
  const sessionCellIds = new Set(
    session
      ? heatmapRois
          .filter((roi) => isCanonicalRoi(roi))
          .map((roi) => `${roi}__${session.comparison.model}`)
      : [],
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
          session={session}
          focusedStimulusId={focusedStimulusId}
          sessionCellIds={sessionCellIds}
          onRankingSystemChange={setRankingSystem}
          onSelectedRankingRoiChange={setSelectedRankingRoi}
          onSortDirectionToggle={() => setSortDirection((currentDirection) => (currentDirection === 'desc' ? 'asc' : 'desc'))}
          onModelSearchChange={setModelSearch}
          onShowScoreLabelsChange={setShowScoreLabels}
          onResetControls={handleResetControls}
          onSelectCell={handleHeatmapCellClick}
          onCompareModeChange={handleCompareModeChange}
          onFocusStimulus={setFocusedStimulusId}
        />
      </section>
      <section className="tab-panel" hidden={activeTab !== 'user-study'} aria-label="User study view">
        <UserStudyView />
      </section>
    </main>
  );
}
