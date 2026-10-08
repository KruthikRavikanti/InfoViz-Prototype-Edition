import { useEffect, useState } from 'react';
import { BrainMapDialog } from './BrainMapDialog';
import { VoxelSelectivityAtlasDialog } from './VoxelSelectivityAtlasDialog';
import { loadSelectivityAtlasManifest } from '../services/selectivityAtlasAssets';
import type { CortexComparisonSession } from '../types/session';

type SpatialToolsProps = {
  model: string;
  session: CortexComparisonSession | null;
  focusedStimulusId: string | null;
  onFocusStimulus: (stimulusId: string) => void;
};

export function SpatialTools({ model, session, focusedStimulusId, onFocusStimulus }: SpatialToolsProps) {
  const [atlasModels, setAtlasModels] = useState<Set<string> | null>(null);
  const [atlasError, setAtlasError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let ignore = false;
    loadSelectivityAtlasManifest(fetch, controller.signal)
      .then((manifest) => {
        if (!ignore) setAtlasModels(new Set(manifest.supportedModels));
      })
      .catch(() => { if (!ignore) setAtlasError(true); });
    return () => { ignore = true; controller.abort(); };
  }, []);

  const atlasAvailable = atlasModels?.has(model) ?? false;

  return (
    <div className="spatial-tools">
      <p className="spatial-tools-intro">Two views of the selected model’s predicted responses.</p>
      <div className="spatial-tool-grid">
        <article className="spatial-tool-card">
          <p className="eyebrow">Where responses are predicted</p>
          <h3>Cortical response map</h3>
          <p>Browse the 185 reference images or your uploads on the NSD Subject 1 cortical surface.</p>
          <BrainMapDialog
            selectedModel={model}
            session={session}
            focusedStimulusId={focusedStimulusId}
            onFocusStimulus={onFocusStimulus}
          />
          <small>FFA, EBA, and PPA only. Predictions, not measured fMRI.</small>
        </article>
        <article className="spatial-tool-card">
          <p className="eyebrow">How voxels respond across images</p>
          <h3>Voxel response profiles</h3>
          <p>Explore voxels grouped by their predicted response patterns across the reference set.</p>
          <VoxelSelectivityAtlasDialog model={model} available={atlasAvailable} />
          <small>
            {atlasError ? 'Voxel selectivity availability could not be loaded.'
              : atlasModels === null ? 'Checking model availability…'
              : !atlasAvailable ? 'Voxel selectivity is not available for this model.'
              : 'Reference images only; uploads do not refit this atlas.'}
          </small>
        </article>
      </div>
    </div>
  );
}
