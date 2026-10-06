import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { BrainMapHoverDatum } from '../types/brainMap';
import { SUBJECT_ONE_ROI_VOXEL_COUNTS } from '../types/session';
import type { BrainMapImage, BrainMapImageSource } from '../utils/brainMapSelection';
import { buildBrainVoxelImagePeaks } from '../utils/brainVoxelImagePeaks';
import { BrainMapTooltip } from './BrainMapTooltip';

const hovered: BrainMapHoverDatum = {
  hemisphere: 'rh', vertexIndex: 47394, surfaceResponse: 0.456,
  overlappingRois: ['eba', 'ppa'],
  dominantSource: { roi: 'eba', responseIndex: 5, response: -0.4, mappingWeight: 0.75 },
  volumeCoordinateZyx: [63, 29, 45], volumeLinearIndexXyz: 1245, canvasX: 340, canvasY: 400,
};

function image(id: string, source: BrainMapImageSource, response: number): BrainMapImage {
  const roiResponses = {
    ffa: Array.from({ length: SUBJECT_ONE_ROI_VOXEL_COUNTS.ffa }, () => 0),
    ppa: Array.from({ length: SUBJECT_ONE_ROI_VOXEL_COUNTS.ppa }, () => 0),
    eba: Array.from({ length: SUBJECT_ONE_ROI_VOXEL_COUNTS.eba }, () => 0),
  };
  roiResponses.eba[5] = response;
  return { id, source, displayName: `${id}.png`, imageUrl: `/images/${id}.png`, result: { stimulusId: id, roiResponses } };
}

function render(images: BrainMapImage[], source: BrainMapImageSource) {
  return renderToStaticMarkup(<BrainMapTooltip
    hovered={hovered}
    imagePeaks={buildBrainVoxelImagePeaks(images, source)}
    displayedImageName="currently-displayed.png"
  />);
}

describe('brain voxel highest-response image preview', () => {
  it('adds an uploaded-image preview while preserving all existing hover information', () => {
    const markup = render([
      image('upload-a', 'uploaded', -0.4), image('upload-b', 'uploaded', 0.7), image('reference', 'reference', 100),
    ], 'uploaded');
    expect(markup).toContain('Highest-response image for this source voxel');
    expect(markup).toContain('Among 2 uploads');
    expect(markup).toContain('data-peak-image-id="upload-b"');
    expect(markup).toContain('data-peak-image-source="uploaded"');
    expect(markup).toContain('alt="upload-b.png"');
    expect(markup).toContain('Highest predicted response: 0.700');
    expect(markup).not.toContain('alt="reference.png"');
    for (const detail of [
      'EBA · RH', 'Displayed image: currently-displayed.png', 'Surface value: 0.456',
      'Dominant source response: -0.400', 'ROI response index: 5',
      'Volume voxel [z, y, x]: [63, 29, 45]', 'Surface vertex: 47,394', 'ROI overlap: EBA, PPA',
    ]) expect(markup).toContain(detail);
    expect(markup).not.toContain('<button'); // Hover never auto-selects the winning image.
  });

  it('only shows reference images when the reference collection is active', () => {
    const markup = render([image('upload', 'uploaded', 20), image('reference-a', 'reference', 0.5)], 'reference');
    expect(markup).toContain('Among 1 Murty-185 reference images');
    expect(markup).toContain('data-peak-image-id="reference-a"');
    expect(markup).toContain('data-peak-image-source="reference"');
    expect(markup).not.toContain('alt="upload.png"');
  });

  it('qualifies negative maxima and exact ties without calling them measured activation', () => {
    const markup = render([
      image('first', 'uploaded', -0.2), image('second', 'uploaded', -0.2), image('lowest', 'uploaded', -3),
    ], 'uploaded');
    expect(markup).toContain('data-peak-image-id="first"');
    expect(markup).toContain('data-peak-tie-count="2"');
    expect(markup).toContain('Highest predicted response: -0.200');
    expect(markup).toContain('Tied highest: 2 images; first shown.');
    expect(markup).toContain('All predictions here are negative; this is the least negative.');
    expect(markup).not.toContain('measured activation');
  });

  it('keeps voxel information but does not invent a preview if ranking is unavailable', () => {
    const markup = render([], 'uploaded');
    expect(markup).toContain('No valid image ranking is available for this source voxel.');
    expect(markup).toContain('Surface value: 0.456');
    expect(markup).not.toContain('data-peak-image-id=');
    expect(markup).not.toContain('<img');
  });
});
