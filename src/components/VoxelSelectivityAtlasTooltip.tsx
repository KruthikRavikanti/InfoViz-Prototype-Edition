import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type { SelectivityAtlasHoverDatum } from '../types/selectivityAtlas';
import { brainTooltipPosition } from '../utils/brainTooltipLayout';

export function VoxelSelectivityAtlasTooltip({ hovered }: { hovered: SelectivityAtlasHoverDatum }) {
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const latestHoverRef = useRef(hovered);
  const [position, setPosition] = useState({ left: 0, top: 0 });

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
    <div
      ref={tooltipRef}
      className="selectivity-atlas-tooltip"
      style={position}
      role="status"
      data-voxel-index={hovered.voxelIndex}
    >
      <strong>{hovered.roi.toUpperCase()} voxel {hovered.roiVoxelIndex}</strong>
      <span>Cluster C{hovered.clusterLabel}</span>
      <span>PCA centroid distance {hovered.distanceToCentroid.toFixed(3)}</span>
      {hovered.outsideView && <span>Outside current view · zoom out or reset.</span>}
    </div>
  );
}
