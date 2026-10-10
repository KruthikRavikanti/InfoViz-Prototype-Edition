import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type { BrainMapHoverDatum } from '../types/brainMap';
import { brainTooltipPosition } from '../utils/brainTooltipLayout';
import { getBrainVoxelImagePeak, type BrainVoxelImagePeakIndex } from '../utils/brainVoxelImagePeaks';
import type { BrainMapImage } from '../utils/brainMapSelection';

function formatResponse(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

type ImagePreview = Pick<BrainMapImage, 'id' | 'displayName' | 'imageUrl'>;

function BrainHoverImage({ image }: { image: ImagePreview }) {
  const [failedImageUrl, setFailedImageUrl] = useState<string | null>(null);
  return failedImageUrl === image.imageUrl ? (
    <span className="brain-map-voxel-peak-missing">Preview unavailable</span>
  ) : (
    <img
      src={image.imageUrl}
      alt={image.displayName}
      width={64}
      height={64}
      decoding="async"
      onError={() => setFailedImageUrl(image.imageUrl)}
    />
  );
}

export function BrainMapTooltip({ hovered, imagePeaks, displayedImage, onDismiss }: {
  hovered: BrainMapHoverDatum;
  imagePeaks: BrainVoxelImagePeakIndex;
  displayedImage: ImagePreview;
  onDismiss?: () => void;
}) {
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const latestHoverRef = useRef(hovered);
  const [position, setPosition] = useState({ left: 0, top: 0 });
  const [isScrollable, setIsScrollable] = useState(false);
  const peak = getBrainVoxelImagePeak(imagePeaks, hovered.dominantSource.roi, hovered.dominantSource.responseIndex);
  const collectionLabel = imagePeaks.source === 'uploaded' ? 'uploads' : 'Murty-185 reference images';

  const updatePosition = useCallback(() => {
    const tooltip = tooltipRef.current;
    const stage = tooltip?.parentElement;
    if (!tooltip || !stage) return;
    const stageBounds = stage.getBoundingClientRect();
    const tooltipBounds = tooltip.getBoundingClientRect();
    setIsScrollable(tooltip.scrollHeight > tooltip.clientHeight);
    const nextPosition = brainTooltipPosition({
      canvasX: latestHoverRef.current.canvasX,
      canvasY: latestHoverRef.current.canvasY,
      stageWidth: stageBounds.width,
      stageHeight: stageBounds.height,
      tooltipWidth: tooltipBounds.width,
      tooltipHeight: tooltipBounds.height,
    });
    setPosition((current) => current.left === nextPosition.left && current.top === nextPosition.top
      ? current : nextPosition);
  }, []);

  useLayoutEffect(() => {
    latestHoverRef.current = hovered;
    updatePosition();
  }, [hovered, updatePosition]);

  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    const stage = tooltip?.parentElement;
    if (!tooltip || !stage) return;
    const observer = new ResizeObserver(updatePosition);
    observer.observe(stage);
    observer.observe(tooltip);
    return () => observer.disconnect();
  }, [updatePosition]);

  return (
    <div
      ref={tooltipRef}
      className="brain-map-tooltip"
      style={position}
      role="status"
      tabIndex={0}
      aria-label="Voxel response comparison and anatomical details"
      onPointerLeave={(event) => {
        if (!event.currentTarget.contains(document.activeElement)) onDismiss?.();
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget) && !event.currentTarget.matches(':hover')) onDismiss?.();
      }}
    >
      <strong>{hovered.dominantSource.roi.toUpperCase()} · {hovered.hemisphere.toUpperCase()}</strong>
      <span className="brain-map-tooltip-collection">Among {imagePeaks.candidateCount} {collectionLabel} · model units</span>
      <div className="brain-map-hover-comparison">
        <div className="brain-map-hover-image brain-map-hover-displayed" data-displayed-image-id={displayedImage.id}>
          <strong className="brain-map-context-label">Shown on brain</strong>
          <BrainHoverImage image={displayedImage} />
          <strong className="brain-map-tooltip-displayed-image">{displayedImage.displayName}</strong>
          <span>Dominant source response: {formatResponse(hovered.dominantSource.response)}</span>
        </div>
        <div
          className="brain-map-hover-image brain-map-voxel-peak"
          data-peak-image-id={peak?.image.id}
          data-peak-image-source={imagePeaks.source}
          data-peak-response={peak?.response}
          data-peak-tie-count={peak?.tieCount}
        >
          <strong className="brain-map-context-label">Highest at this voxel</strong>
          {peak ? (
            <>
              <BrainHoverImage image={peak.image} />
              <strong>{peak.image.displayName}</strong>
              <span>Highest predicted response: {formatResponse(peak.response)}</span>
            </>
          ) : (
            <span>No valid image ranking is available for this source voxel.</span>
          )}
        </div>
      </div>
      {peak ? (
        <span className="brain-map-hover-context-note">
          {peak.image.id === displayedImage.id
            ? 'Same image as brain map' : 'Comparison preview · brain unchanged'}
        </span>
      ) : null}
      {peak && peak.tieCount > 1 ? <span>Tied highest: {peak.tieCount} images; first shown.</span> : null}
      {peak && peak.response < 0 ? <span className="brain-map-voxel-peak-note">All predictions here are negative; this is the least negative.</span> : null}
      {isScrollable ? <span className="brain-map-hover-scroll-hint">Scroll for voxel details ↓</span> : null}
      <div className="brain-map-hover-geometry">
        <span>Selected image surface value: {formatResponse(hovered.surfaceResponse)}</span>
        <span>ROI response index: {hovered.dominantSource.responseIndex}</span>
        <span>Volume voxel [z, y, x]: [{hovered.volumeCoordinateZyx.join(', ')}]</span>
        <span>Surface vertex: {hovered.vertexIndex.toLocaleString()}</span>
        {hovered.overlappingRois.length > 1 && (
          <span>ROI overlap: {hovered.overlappingRois.map((roi) => roi.toUpperCase()).join(', ')}</span>
        )}
      </div>
    </div>
  );
}
