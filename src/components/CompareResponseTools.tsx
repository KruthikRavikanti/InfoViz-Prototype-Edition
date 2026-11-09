import { useMemo, useState } from 'react';
import type { ClusteringData, EvidenceImage, EvidenceView, SelectedHeatmapCell } from '../types/data';
import { buildSharedResponseComparison } from '../utils/responseComparison';
import { CompareResponseScatter, comparisonModelLabel } from './CompareResponseScatter';
import { ComparisonPinMarker } from './ComparisonPinMarker';
import { LinkedComparisonClusters } from './LinkedComparisonClusters';

type Props = {
  cellA: SelectedHeatmapCell;
  cellB: SelectedHeatmapCell;
  evidenceA: EvidenceView | null;
  evidenceB: EvidenceView | null;
  clustering: ClusteringData;
  onOpenImage: (image: EvidenceImage) => void;
};

export function CompareResponseTools({ cellA, cellB, evidenceA, evidenceB, clustering, onOpenImage }: Props) {
  const [activeView, setActiveView] = useState<'responses' | 'clusters'>('responses');
  const [clustersVisited, setClustersVisited] = useState(false);
  const [pinnedNames, setPinnedNames] = useState<string[]>([]);
  const [notice, setNotice] = useState('');
  const comparison = useMemo(() => buildSharedResponseComparison(evidenceA?.images ?? [], evidenceB?.images ?? []), [evidenceA, evidenceB]);
  const pointsByName = useMemo(() => new Map(comparison.points.map(point => [point.imageName, point])), [comparison.points]);
  const pinnedImages = pinnedNames.flatMap(name => {
    const image = pointsByName.get(name);
    return image ? [image] : [];
  });

  function togglePin(imageName: string) {
    if (!pointsByName.has(imageName)) return;
    if (pinnedNames.includes(imageName)) {
      setPinnedNames(current => current.filter(name => name !== imageName));
      setNotice('');
    } else if (pinnedNames.length < 2) {
      setPinnedNames(current => [...current, imageName]);
      setNotice('');
    } else {
      setNotice('Two stimuli are pinned. Remove one to choose another.');
    }
  }

  function choosePin(index: number, name: string) {
    if (!name) setPinnedNames(current => current.filter((_, i) => i !== index));
    else if (pointsByName.has(name)) setPinnedNames(current => {
      const next = [...current]; next[index] = name;
      return next.filter(Boolean);
    });
    setNotice('');
  }

  return <section className="compare-response-tools" aria-label="Linked image comparison">
    <div className="compare-linked-tabs" role="tablist" aria-label="Image comparison views">
      <button type="button" role="tab" id="compare-responses-tab" aria-selected={activeView === 'responses'}
        aria-controls="compare-linked-panel" onClick={() => setActiveView('responses')}>Predicted responses</button>
      <button type="button" role="tab" id="compare-clusters-tab" aria-selected={activeView === 'clusters'}
        aria-controls="compare-linked-panel" onClick={() => { setClustersVisited(true); setActiveView('clusters'); }}>Pinned image clusters</button>
    </div>
    <div className="compare-pin-heading"><h3>Pinned stimuli</h3><p>Choose up to two images, or select points in the scatterplot. #1 is the highest predicted response.</p></div>
    <div className="compare-pins" aria-label="Pinned stimuli">
      {[0, 1].map(index => {
        const image = pinnedImages[index];
        return <div className="compare-pin-card" key={index} data-pin-slot={index + 1} data-pinned-image={image?.imageId}>
          <svg viewBox="0 0 30 30" aria-hidden="true"><ComparisonPinMarker index={index} x={15} y={15} /></svg>
          {image && <img src={image.imageUrl} alt={image.imageId} />}
          <div className="compare-pin-content">
            <label>Stimulus {index + 1}
              <select aria-label={`Pinned stimulus ${index + 1}`} value={image?.imageName ?? ''}
                disabled={Boolean(comparison.unavailableReason) || (index === 1 && pinnedImages.length === 0)}
                onChange={event => choosePin(index, event.target.value)}>
                <option value="">Choose reference image…</option>
                {comparison.points.map(point => <option key={point.imageName} value={point.imageName}
                  disabled={pinnedNames.includes(point.imageName) && image?.imageName !== point.imageName}>{point.imageName}</option>)}
              </select>
            </label>
            {image && <div className="compare-pin-ranks">
              <span>{comparisonModelLabel(cellA.model)} <strong>#{image.rankA}</strong></span>
              <span>{comparisonModelLabel(cellB.model)} <strong>#{image.rankB}</strong></span>
            </div>}
          </div>
          {image && <button type="button" className="compare-unpin" aria-label={`Unpin ${image.imageId}`} onClick={() => togglePin(image.imageName)}>×</button>}
        </div>;
      })}
    </div>
    <p className="compare-pin-notice" role="status">{notice}</p>
    <div id="compare-linked-panel" role="tabpanel" aria-labelledby={activeView === 'responses' ? 'compare-responses-tab' : 'compare-clusters-tab'}>
      {comparison.unavailableReason ? <p className="compare-empty">{comparison.unavailableReason}</p> : <>
        <div hidden={activeView !== 'responses'}><CompareResponseScatter points={comparison.points} cellA={cellA} cellB={cellB}
          pinnedImages={pinnedImages} onTogglePin={togglePin} /></div>
        {clustersVisited && <div hidden={activeView !== 'clusters'}><LinkedComparisonClusters cellA={cellA} cellB={cellB} clustering={clustering}
          pinnedImages={pinnedImages} onOpenImage={onOpenImage} /></div>}
      </>}
    </div>
  </section>;
}
