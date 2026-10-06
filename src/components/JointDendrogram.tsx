import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { ClusterView, EvidenceImage } from '../types/data';
import type { CortexComparisonSession, JointHierarchy } from '../types/session';

type LinkedItem = { kind: 'reference' | 'uploaded'; id: string; label: string };

type JointDendrogramProps = {
  view: ClusterView;
  session: CortexComparisonSession;
  onOpenImage: (image: EvidenceImage) => void;
  linkedItem?: LinkedItem | null;
  onLinkedItemChange?: (item: LinkedItem | null) => void;
};

type TreeNode = {
  x: number;
  distance: number;
  size: number;
  clusterId: string | null;
};

type TreeLayout = {
  width: number;
  height: number;
  baseline: number;
  maxDistance: number;
  leafOrder: number[];
  nodes: TreeNode[];
};

const CLUSTER_COLORS = ['#4F5FA6', '#C0392B', '#2A7D5B', '#B45309', '#6B5EA8', '#1A7FA3', '#8B5E3C', '#3D7A6A'];
const LEAF_SPACING = 20;
const LEFT_PADDING = 66;
const TOP_PADDING = 22;
const BASELINE = 236;

/** Render only complete binary linkage trees; malformed input cannot invent leaves. */
export function buildJointDendrogramLayout(hierarchy: JointHierarchy): TreeLayout | null {
  const count = hierarchy.members.length;
  if (count < 2 || hierarchy.linkage.length !== count - 1) return null;
  const sizes = Array<number>(count).fill(1);
  const used = new Set<number>();
  let maxDistance = 0;
  for (let rowIndex = 0; rowIndex < hierarchy.linkage.length; rowIndex += 1) {
    const [left, right, distance, size] = hierarchy.linkage[rowIndex];
    const nodeIndex = count + rowIndex;
    if (!Number.isInteger(left) || !Number.isInteger(right) || left < 0 || right < 0
      || left >= nodeIndex || right >= nodeIndex || left === right
      || used.has(left) || used.has(right)
      || !Number.isFinite(distance) || distance < 0 || distance > 2 + 1e-7
      || distance < maxDistance - 1e-12
      || size !== sizes[left] + sizes[right]) return null;
    used.add(left);
    used.add(right);
    sizes.push(size);
    maxDistance = Math.max(maxDistance, distance);
  }
  if (sizes[sizes.length - 1] !== count || used.size !== 2 * count - 2) return null;

  const leafOrder: number[] = [];
  const stack = [2 * count - 2];
  while (stack.length > 0) {
    const nodeIndex = stack.pop()!;
    if (nodeIndex < count) leafOrder.push(nodeIndex);
    else {
      const [left, right] = hierarchy.linkage[nodeIndex - count];
      stack.push(right, left);
    }
  }
  const xByIndex = new Map(leafOrder.map((index, order) => [index, LEFT_PADDING + order * LEAF_SPACING]));
  const nodes: TreeNode[] = hierarchy.members.map((member, index) => ({
    x: xByIndex.get(index)!, distance: 0, size: 1, clusterId: String(member.clusterId),
  }));
  for (const [left, right, distance, size] of hierarchy.linkage) {
    nodes.push({
      x: (nodes[left].x + nodes[right].x) / 2,
      distance,
      size,
      clusterId: nodes[left].clusterId === nodes[right].clusterId ? nodes[left].clusterId : null,
    });
  }
  return {
    width: Math.max(800, LEFT_PADDING + (count - 1) * LEAF_SPACING + 30),
    height: 350,
    baseline: BASELINE,
    maxDistance: Math.max(.1, Math.min(2, maxDistance * 1.05)),
    leafOrder,
    nodes,
  };
}

function colorForCluster(clusterId: string): string {
  const index = Number(clusterId);
  return CLUSTER_COLORS[Number.isInteger(index) && index >= 0 ? index % CLUSTER_COLORS.length : 0];
}

function shortLabel(label: string): string {
  return label.length <= 19 ? label : `${label.slice(0, 16)}…`;
}

export function JointDendrogram({ view, session, onOpenImage, linkedItem = null, onLinkedItemChange }: JointDendrogramProps) {
  const hierarchy = view.jointHierarchy;
  const layout = useMemo(() => hierarchy ? buildJointDendrogramLayout(hierarchy) : null, [hierarchy]);
  const images = useMemo(() => {
    if (!hierarchy) return [];
    const referenceById = new Map(view.points.map((point) => [point.imageName, point]));
    const uploadsById = new Map(session.stimuli.map((stimulus) => [stimulus.id, stimulus]));
    return hierarchy.members.map((member): EvidenceImage | null => {
      const valueLabel = `Cluster ${member.clusterId} · correlation distance to medoid ${member.distanceToMedoid.toFixed(3)}`;
      if (member.kind === 'reference') {
        const point = referenceById.get(member.id);
        return point ? {
          imageName: point.imageName, imageUrl: point.imageUrl, value: member.distanceToMedoid,
          valueLabel, rank: point.rank, origin: 'reference',
        } : null;
      }
      const stimulus = uploadsById.get(member.id);
      return stimulus ? {
        imageName: stimulus.displayName, imageUrl: stimulus.imageUrl, value: member.distanceToMedoid,
        valueLabel: `Uploaded · ${valueLabel}`, rank: 0, origin: 'uploaded', stimulusId: stimulus.id,
        markerColor: '#1c1917',
      } : null;
    });
  }, [hierarchy, view.points, session.stimuli]);
  const [preview, setPreview] = useState<{
    memberIndex: number; trigger: 'hover' | 'focus'; anchorX: number; anchorY: number;
  } | null>(null);
  const tooltipId = useId();
  const tooltipRef = useRef<HTMLDivElement>(null);
  const linkedCallbackRef = useRef(onLinkedItemChange);
  const previewVisible = preview !== null;

  useEffect(() => { linkedCallbackRef.current = onLinkedItemChange; }, [onLinkedItemChange]);
  useEffect(() => {
    setPreview(null);
    linkedCallbackRef.current?.(null);
  }, [hierarchy]);
  useEffect(() => () => { linkedCallbackRef.current?.(null); }, []);

  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    if (!preview || !tooltip) return;
    const height = tooltip.offsetHeight;
    const preferredTop = preview.anchorY - height - 10 >= 8
      ? preview.anchorY - height - 10 : preview.anchorY + 10;
    tooltip.style.top = `${Math.max(8, Math.min(window.innerHeight - height - 8, preferredTop))}px`;
  }, [preview]);

  useEffect(() => {
    if (!previewVisible) return;
    const dismiss = () => {
      setPreview(null);
      onLinkedItemChange?.(null);
    };
    window.addEventListener('scroll', dismiss, { capture: true, passive: true });
    window.addEventListener('resize', dismiss);
    return () => {
      window.removeEventListener('scroll', dismiss, true);
      window.removeEventListener('resize', dismiss);
    };
  }, [previewVisible, onLinkedItemChange]);

  if (!hierarchy || !layout || images.some((image) => image === null)) {
    return <p className="cluster-empty-note">The session dendrogram is unavailable.</p>;
  }

  const y = (distance: number) => layout.baseline - (distance / layout.maxDistance) * (layout.baseline - TOP_PADDING);
  const tickValues = Array.from({ length: 5 }, (_, index) => layout.maxDistance * index / 4);
  const previewImage = preview ? images[preview.memberIndex] : null;
  const previewMember = preview ? hierarchy.members[preview.memberIndex] : null;

  function clearPreview() {
    setPreview(null);
    onLinkedItemChange?.(null);
  }

  function showPreview(memberIndex: number, target: SVGGraphicsElement, trigger: 'hover' | 'focus') {
    const bounds = target.getBoundingClientRect();
    const member = hierarchy!.members[memberIndex];
    const image = images[memberIndex]!;
    setPreview({ memberIndex, trigger, anchorX: bounds.left + bounds.width / 2, anchorY: bounds.top + 4 });
    onLinkedItemChange?.({ kind: member.kind, id: member.id, label: image.imageName });
  }

  function openImage(memberIndex: number) {
    clearPreview();
    onOpenImage(images[memberIndex]!);
  }

  return (
    <figure className="joint-dendrogram" data-joint-dendrogram="true">
      <div className="joint-dendrogram-scroll" role="region" aria-label="Joint image dendrogram; scroll horizontally to see all leaves" tabIndex={0}>
        <svg width={layout.width} height={layout.height} viewBox={`0 0 ${layout.width} ${layout.height}`} role="group" aria-label={`Joint hierarchical dendrogram with ${hierarchy.members.length} image leaves`}>
          <text className="joint-dendrogram-axis-label" x={18} y={layout.baseline / 2} transform={`rotate(-90 18 ${layout.baseline / 2})`} textAnchor="middle">Average-linkage correlation distance (1 − r)</text>
          {tickValues.map((distance) => (
            <g key={distance} className="joint-dendrogram-axis-tick" data-correlation-distance={distance}>
              <line x1={LEFT_PADDING - 10} x2={layout.width - 12} y1={y(distance)} y2={y(distance)} />
              <text x={LEFT_PADDING - 15} y={y(distance) + 4} textAnchor="end">{distance.toFixed(2)}</text>
            </g>
          ))}
          {hierarchy.linkage.map(([left, right, distance], rowIndex) => {
            const node = layout.nodes[hierarchy.members.length + rowIndex];
            return (
              <path
                key={rowIndex}
                className="joint-dendrogram-branch"
                data-linkage-node={hierarchy.members.length + rowIndex}
                data-merge-distance={distance}
                fill="none"
                stroke={node.clusterId === null ? 'var(--text-muted)' : colorForCluster(node.clusterId)}
                d={`M ${layout.nodes[left].x} ${y(layout.nodes[left].distance)} V ${y(distance)} H ${layout.nodes[right].x} V ${y(layout.nodes[right].distance)}`}
              />
            );
          })}
          {layout.leafOrder.map((memberIndex) => {
            const member = hierarchy.members[memberIndex];
            const image = images[memberIndex]!;
            const isUploaded = member.kind === 'uploaded';
            const highlighted = preview?.memberIndex === memberIndex || (linkedItem?.kind === member.kind && linkedItem.id === member.id);
            const label = `${isUploaded ? 'Uploaded stimulus' : 'Reference image'} ${image.imageName}, jointly clustered in cluster ${member.clusterId}, correlation distance to medoid ${member.distanceToMedoid.toFixed(3)}`;
            return (
              <g
                key={`${member.kind}:${member.id}`}
                className={`joint-dendrogram-leaf${isUploaded ? ' joint-dendrogram-leaf--uploaded' : ''}${highlighted ? ' joint-dendrogram-leaf--highlighted' : ''}`}
                transform={`translate(${layout.nodes[memberIndex].x},${layout.baseline})`}
                data-joint-leaf={member.kind}
                data-link-key={`${member.kind}:${member.id}`}
                data-cluster-id={member.clusterId}
                role="button"
                tabIndex={0}
                aria-label={label}
                aria-describedby={preview?.memberIndex === memberIndex ? tooltipId : undefined}
                onMouseEnter={(event) => showPreview(memberIndex, event.currentTarget, 'hover')}
                onMouseLeave={() => { if (preview?.trigger !== 'focus') clearPreview(); }}
                onFocus={(event) => showPreview(memberIndex, event.currentTarget, 'focus')}
                onBlur={clearPreview}
                onClick={() => openImage(memberIndex)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    openImage(memberIndex);
                  } else if (event.key === 'Escape') {
                    event.preventDefault();
                    event.stopPropagation();
                    clearPreview();
                  }
                }}
              >
                <rect x={-9} y={-6} width={18} height={109} fill="transparent" />
                {isUploaded && <circle className="joint-dendrogram-upload-glow" r={6} fill="#1c1917" opacity={.15} />}
                <circle className="joint-dendrogram-leaf-dot" r={highlighted ? 4.5 : 3} fill={isUploaded ? '#1c1917' : colorForCluster(String(member.clusterId))} />
                <text className="joint-dendrogram-leaf-label" x={4} y={13} transform="rotate(65 4 13)">{shortLabel(image.imageName)}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <figcaption>Each leaf is an image. Hover to preview; click to enlarge. Glowing leaves are uploads.</figcaption>
      {preview && previewImage && previewMember && createPortal(
        <div
          id={tooltipId}
          ref={tooltipRef}
          className="scatter-dot-tooltip scatter-dot-tooltip--uploaded joint-dendrogram-tooltip"
          role="tooltip"
          style={{ left: `clamp(74px, ${preview.anchorX}px, calc(100vw - 74px))`, top: `${preview.anchorY}px` }}
        >
          <img className="scatter-dot-tooltip-img" src={previewImage.imageUrl} alt="" />
          <p className="scatter-dot-tooltip-label">{previewImage.imageName}</p>
          <p className="scatter-dot-tooltip-sub scatter-dot-tooltip-sub--uploaded">
            <span>{previewMember.kind === 'uploaded' ? 'Uploaded · ' : ''}Cluster {previewMember.clusterId}</span>
            <span>Joint hierarchical clustering</span>
            <span>Correlation distance to medoid {previewMember.distanceToMedoid.toFixed(3)}</span>
          </p>
        </div>,
        document.body,
      )}
    </figure>
  );
}
