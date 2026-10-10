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

const displayedImage: Pick<BrainMapImage, 'id' | 'displayName' | 'imageUrl'> = {
  id: 'currently-displayed',
  displayName: 'currently-displayed.png',
  imageUrl: '/images/currently-displayed.png',
};

function render(
  images: BrainMapImage[],
  source: BrainMapImageSource,
  currentImage = displayedImage,
) {
  return renderToStaticMarkup(<BrainMapTooltip
    hovered={hovered}
    imagePeaks={buildBrainVoxelImagePeaks(images, source)}
    displayedImage={currentImage}
  />);
}

describe('brain voxel highest-response image preview', () => {
  it('adds an uploaded-image preview while preserving all existing hover information', () => {
    const markup = render([
      image('upload-a', 'uploaded', -0.4), image('upload-b', 'uploaded', 0.7), image('reference', 'reference', 100),
    ], 'uploaded');
    expect(markup).toContain('Shown on brain');
    expect(markup).toContain('alt="currently-displayed.png"');
    expect(markup).toContain('src="/images/currently-displayed.png"');
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('aria-label="Voxel response comparison and anatomical details"');
    expect(markup).toContain('Highest at this voxel');
    expect(markup).toContain('Comparison preview · brain unchanged');
    expect(markup).not.toContain('Same image as brain map');
    expect(markup).toContain('Among 2 uploads');
    expect(markup).toContain('data-peak-image-id="upload-b"');
    expect(markup).toContain('data-peak-image-source="uploaded"');
    expect(markup).toContain('alt="upload-b.png"');
    expect(markup).toContain('Highest predicted response: 0.700');
    expect(markup).not.toContain('alt="reference.png"');
    for (const detail of [
      'EBA · RH', 'currently-displayed.png', 'Selected image surface value: 0.456',
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

  it('keeps the displayed-image preview and voxel information when ranking is unavailable', () => {
    const markup = render([], 'uploaded');
    expect(markup).toContain('No valid image ranking is available for this source voxel.');
    expect(markup).toContain('Shown on brain');
    expect(markup).toContain('alt="currently-displayed.png"');
    expect(markup).toContain('Selected image surface value: 0.456');
    expect(markup).not.toContain('data-peak-image-id=');
    expect(markup.match(/<img /g)).toHaveLength(1);
    expect(markup).not.toContain('Comparison preview · brain unchanged');
    expect(markup).not.toContain('Same image as brain map');
  });

  it('clearly identifies when the displayed image also has the highest voxel response', () => {
    const currentImage = image('displayed-winner', 'uploaded', -0.4);
    const markup = render([currentImage, image('other', 'uploaded', -0.7)], 'uploaded', currentImage);
    expect(markup).toContain('Shown on brain');
    expect(markup).toContain('Highest at this voxel');
    expect(markup).toContain('data-peak-image-id="displayed-winner"');
    expect(markup).toContain('Same image as brain map');
    expect(markup).not.toContain('Comparison preview · brain unchanged');
    expect(markup).not.toContain('<button');
  });

  it('distinguishes different images with the same display name by their IDs', () => {
    const currentImage = { ...image('displayed', 'uploaded', -0.4), displayName: 'duplicate.png' };
    const winner = { ...image('different-winner', 'uploaded', 0.7), displayName: 'duplicate.png' };
    const markup = render([currentImage, winner], 'uploaded', currentImage);
    expect(markup).toContain('src="/images/displayed.png"');
    expect(markup).toContain('src="/images/different-winner.png"');
    expect(markup).toContain('data-peak-image-id="different-winner"');
    expect(markup).toContain('Comparison preview · brain unchanged');
    expect(markup).not.toContain('Same image as brain map');
  });
});
