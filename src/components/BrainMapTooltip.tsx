import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type { BrainMapHoverDatum } from '../types/brainMap';
import { brainTooltipPosition } from '../utils/brainTooltipLayout';
import { getBrainVoxelImagePeak, type BrainVoxelImagePeakIndex } from '../utils/brainVoxelImagePeaks';

function formatResponse(value: number): string {
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

export function BrainMapTooltip({ hovered, imagePeaks, displayedImageName }: {
  hovered: BrainMapHoverDatum;
  imagePeaks: BrainVoxelImagePeakIndex;
  displayedImageName: string;
}) {
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const latestHoverRef = useRef(hovered);
  const [position, setPosition] = useState({ left: 0, top: 0 });
  const [failedImageUrl, setFailedImageUrl] = useState<string | null>(null);
  const peak = getBrainVoxelImagePeak(imagePeaks, hovered.dominantSource.roi, hovered.dominantSource.responseIndex);
  const collectionLabel = imagePeaks.source === 'uploaded' ? 'uploads' : 'Murty-185 reference images';

  const updatePosition = useCallback(() => {
    const tooltip = tooltipRef.current;
    const stage = tooltip?.parentElement;
    if (!tooltip || !stage) return;
    const stageBounds = stage.getBoundingClientRect();
    const tooltipBounds = tooltip.getBoundingClientRect();
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
    <div ref={tooltipRef} className="brain-map-tooltip" style={position} role="status">
      <strong>{hovered.dominantSource.roi.toUpperCase()} · {hovered.hemisphere.toUpperCase()}</strong>
      <span className="brain-map-tooltip-displayed-image">Displayed image: {displayedImageName}</span>
      <span>Surface value: {formatResponse(hovered.surfaceResponse)}</span>
      <span>Dominant source response: {formatResponse(hovered.dominantSource.response)}</span>
      <span>ROI response index: {hovered.dominantSource.responseIndex}</span>
      <span>Volume voxel [z, y, x]: [{hovered.volumeCoordinateZyx.join(', ')}]</span>
      <span>Surface vertex: {hovered.vertexIndex.toLocaleString()}</span>
      {hovered.overlappingRois.length > 1 && (
        <span>ROI overlap: {hovered.overlappingRois.map((roi) => roi.toUpperCase()).join(', ')}</span>
      )}
      <div
        className="brain-map-voxel-peak"
        data-peak-image-id={peak?.image.id}
        data-peak-image-source={imagePeaks.source}
        data-peak-response={peak?.response}
        data-peak-tie-count={peak?.tieCount}
      >
        <strong>Highest-response image for this source voxel</strong>
        <span>Among {imagePeaks.candidateCount} {collectionLabel}</span>
        {peak ? (
          <>
            <div className="brain-map-voxel-peak-preview">
              {failedImageUrl === peak.image.imageUrl ? (
                <span className="brain-map-voxel-peak-missing">Preview unavailable</span>
              ) : (
                <img
                  src={peak.image.imageUrl}
                  alt={peak.image.displayName}
                  width={72}
                  height={72}
                  decoding="async"
                  onError={() => setFailedImageUrl(peak.image.imageUrl)}
                />
              )}
              <div>
                <strong>{peak.image.displayName}</strong>
                <span>Highest predicted response: {formatResponse(peak.response)}</span>
                {peak.tieCount > 1 && <span>Tied highest: {peak.tieCount} images; first shown.</span>}
              </div>
            </div>
            {peak.response < 0 && <span className="brain-map-voxel-peak-note">All predictions here are negative; this is the least negative.</span>}
          </>
        ) : (
          <span>No valid image ranking is available for this source voxel.</span>
        )}
      </div>
    </div>
  );
}
