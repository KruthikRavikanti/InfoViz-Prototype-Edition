import type { BrainMapReferenceSet } from '../types/brainMap';
import { BRAIN_REFERENCE_VERSION } from '../services/brainReferenceAssets';
import { CANONICAL_ROIS, SUBJECT_ONE_ROI_VOXEL_COUNTS, type BrainMapStimulusResult, type CortexComparisonSession } from '../types/session';

export type BrainMapImageSource = 'uploaded' | 'reference';

export type BrainMapImage = {
  id: string;
  source: BrainMapImageSource;
  displayName: string;
  imageUrl: string;
  result: BrainMapStimulusResult;
};

/** A session's vectors cannot be displayed under another model or reference bundle. */
export function brainMapUploadedImages(
  session: CortexComparisonSession | null | undefined,
  model: string,
  references?: BrainMapReferenceSet,
): BrainMapImage[] {
  if (
    !session?.brainMapResults
    || session.comparison.model !== model
    || session.comparison.referenceVersion !== BRAIN_REFERENCE_VERSION
    || session.brainMapResults.schemaVersion !== 1
    || session.brainMapResults.subjectId !== 's1'
    || session.brainMapResults.responseSpace !== 'nsd_1000_subject1_floc_t7_v1'
    || CANONICAL_ROIS.some((roi) => session.brainMapResults!.roiVoxelCounts[roi] !== SUBJECT_ONE_ROI_VOXEL_COUNTS[roi])
    || (references && references.model !== model)
    || (references && references.referenceVersion !== session.comparison.referenceVersion)
  ) return [];

  const results = new Map(session.brainMapResults.stimuli.map((result) => [result.stimulusId, result]));
  return session.stimuli.flatMap((stimulus) => {
    const result = results.get(stimulus.id);
    return result ? [{
      id: stimulus.id,
      source: 'uploaded' as const,
      displayName: stimulus.displayName,
      imageUrl: stimulus.imageUrl,
      result,
    }] : [];
  });
}

export function brainMapReferenceImages(references: BrainMapReferenceSet | undefined): BrainMapImage[] {
  return references?.stimuli.map((stimulus) => ({
    id: stimulus.imageName,
    source: 'reference',
    displayName: stimulus.imageName,
    imageUrl: stimulus.imageUrl,
    result: { stimulusId: `reference:${stimulus.imageName}`, roiResponses: stimulus.roiResponses },
  })) ?? [];
}

export function filterBrainMapImages(images: BrainMapImage[], search: string): BrainMapImage[] {
  const query = search.trim().toLocaleLowerCase();
  return query ? images.filter((image) => image.displayName.toLocaleLowerCase().includes(query)) : images;
}

export function brainMapImageNeighbors(images: BrainMapImage[], selectedId: string | undefined) {
  const index = images.findIndex((image) => image.id === selectedId);
  return {
    previous: index > 0 ? images[index - 1] : undefined,
    next: index < images.length - 1 ? images[index + 1] : undefined,
  };
}
