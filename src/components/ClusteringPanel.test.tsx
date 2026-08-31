import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { ClusteringPanel } from './ClusteringPanel';
import { CLUSTERING_METHODS } from '../utils/clustering';
import { validateComparisonSession } from '../services/cortexSession';
import { rawSessionFixture } from '../test/sessionFixture';

const commonProps = {
  voxelView: null,
  voxelStatus: 'idle' as const,
  clusteringMethod: 'kmeans_euclidean' as const,
  clusteringMethodOptions: CLUSTERING_METHODS,
  onClusteringMethodChange: vi.fn(),
  visualView: null,
  dreamsimView: null,
  imageCategories: new Map(),
  groundTruthPatients: [],
  onOpenImage: vi.fn(),
  selectedRoi: 'ffa',
  focusedStimulusId: 'stimulus-a',
};

describe('session clustering safeguards', () => {
  it('does not claim measured ground-truth data for uploaded stimuli', () => {
    const session = validateComparisonSession(rawSessionFixture(), 'http://core.example:5556');
    const markup = renderToStaticMarkup(<ClusteringPanel {...commonProps} session={session} />);

    expect(markup).toContain('data-no-upload-overlay="true"');
    expect(markup).toContain('not shown in this measured fMRI view');
    expect(markup).toContain('data-dreamsim-overlay="disabled"');
    expect(markup).toContain('1,792 features');
  });

  it('leaves standalone clustering free of session-only notes', () => {
    const markup = renderToStaticMarkup(<ClusteringPanel {...commonProps} session={null} />);
    expect(markup).not.toContain('data-no-upload-overlay="true"');
    expect(markup).not.toContain('data-dreamsim-overlay="disabled"');
  });
});
