import type { ClusterPoint, ClusterView, EvidenceImage } from '../types/data';
import type { CortexComparisonSession, JointHierarchy } from '../types/session';

/** The medoid describes a joint cluster; it is not used to assign its members. */
export function jointClusterMedoidName(
  hierarchy: JointHierarchy,
  clusterId: string,
  session: CortexComparisonSession,
): string | undefined {
  const clusterIndex = Number(clusterId);
  if (!Number.isInteger(clusterIndex) || clusterIndex < 0) return undefined;
  const member = hierarchy.members[hierarchy.medoidIndices[clusterIndex]];
  if (!member || String(member.clusterId) !== clusterId) return undefined;
  return member.kind === 'reference'
    ? member.id
    : session.stimuli.find((stimulus) => stimulus.id === member.id)?.displayName;
}

/**
 * Relabel the fixed reference projection using the hierarchy fitted to the whole
 * session. Uploads remain separate scatter overlays, but count as real members
 * in groups, representative previews, and the dendrogram.
 */
export function buildJointClusterView(
  referenceView: ClusterView | null,
  hierarchy: JointHierarchy | undefined,
  session: CortexComparisonSession,
): ClusterView | null {
  if (!referenceView || !hierarchy) return null;
  if (referenceView.summary?.model && referenceView.summary.model !== session.comparison.model) return null;
  const total = hierarchy.referenceCount + hierarchy.uploadedCount;
  if (hierarchy.referenceCount !== referenceView.points.length
    || hierarchy.uploadedCount !== session.stimuli.length
    || hierarchy.members.length !== total
    || hierarchy.clusterCount < 1
    || !Number.isInteger(hierarchy.clusterCount)
    || hierarchy.medoidIndices.length !== hierarchy.clusterCount) return null;

  const referenceById = new Map(referenceView.points.map((point) => [point.imageName, point]));
  const uploadsById = new Map(session.stimuli.map((stimulus) => [stimulus.id, stimulus]));
  if (referenceById.size !== referenceView.points.length || uploadsById.size !== session.stimuli.length) return null;

  const references = new Map<string, { clusterLabel: string; distance: number }>();
  const uploadIds = new Set<string>();
  const clusterSizes: Record<string, number> = {};
  const imagesByLabel = new Map<string, Array<{ image: EvidenceImage; memberIndex: number }>>();

  for (let memberIndex = 0; memberIndex < hierarchy.members.length; memberIndex += 1) {
    const member = hierarchy.members[memberIndex];
    const label = String(member.clusterId);
    const clusterIndex = Number(label);
    if (!Number.isInteger(clusterIndex) || clusterIndex < 0 || clusterIndex >= hierarchy.clusterCount
      || String(clusterIndex) !== label
      || !Number.isFinite(member.distanceToMedoid)
      || member.distanceToMedoid < 0 || member.distanceToMedoid > 2 + 1e-7) return null;

    let image: EvidenceImage;
    if (member.kind === 'reference') {
      const point = referenceById.get(member.id);
      if (!point || references.has(member.id)) return null;
      references.set(member.id, { clusterLabel: label, distance: member.distanceToMedoid });
      image = {
        imageName: point.imageName,
        imageUrl: point.imageUrl,
        value: member.distanceToMedoid,
        valueLabel: `Correlation distance to medoid ${member.distanceToMedoid.toFixed(3)}`,
        rank: point.rank,
        origin: 'reference',
      };
    } else if (member.kind === 'uploaded') {
      const stimulus = uploadsById.get(member.id);
      if (!stimulus || uploadIds.has(member.id)) return null;
      uploadIds.add(member.id);
      image = {
        imageName: stimulus.displayName,
        imageUrl: stimulus.imageUrl,
        value: member.distanceToMedoid,
        valueLabel: `Uploaded · correlation distance to medoid ${member.distanceToMedoid.toFixed(3)}`,
        rank: 0,
        origin: 'uploaded',
        stimulusId: stimulus.id,
        markerColor: '#1c1917',
      };
    } else return null;

    clusterSizes[label] = (clusterSizes[label] ?? 0) + 1;
    const images = imagesByLabel.get(label) ?? [];
    images.push({ image, memberIndex });
    imagesByLabel.set(label, images);
  }

  if (references.size !== referenceById.size || uploadIds.size !== uploadsById.size
    || imagesByLabel.size !== hierarchy.clusterCount) return null;
  for (let clusterIndex = 0; clusterIndex < hierarchy.clusterCount; clusterIndex += 1) {
    const medoidIndex = hierarchy.medoidIndices[clusterIndex];
    const medoid = hierarchy.members[medoidIndex];
    if (!Number.isInteger(medoidIndex) || !medoid || String(medoid.clusterId) !== String(clusterIndex)) return null;
  }

  const points: ClusterPoint[] = referenceView.points.map((point) => {
    const assigned = references.get(point.imageName)!;
    return {
      ...point,
      clusterLabel: assigned.clusterLabel,
      distanceToCentroid: assigned.distance,
      value: assigned.distance,
    };
  });
  const pointsByLabel = new Map<string, ClusterPoint[]>();
  for (const point of points) {
    const clusterPoints = pointsByLabel.get(point.clusterLabel) ?? [];
    clusterPoints.push(point);
    pointsByLabel.set(point.clusterLabel, clusterPoints);
  }
  const groups = Array.from(imagesByLabel, ([label, candidates]) => ({
    label,
    size: clusterSizes[label],
    points: pointsByLabel.get(label) ?? [],
    images: candidates.slice().sort((a, b) => {
      const medoidIndex = hierarchy.medoidIndices[Number(label)];
      if (a.memberIndex === medoidIndex) return -1;
      if (b.memberIndex === medoidIndex) return 1;
      return a.image.value - b.image.value || a.memberIndex - b.memberIndex;
    }).slice(0, 6).map(({ image }) => image),
  })).sort((a, b) => Number(a.label) - Number(b.label));

  return {
    ...referenceView,
    points,
    groups,
    summary: {
      ...referenceView.summary,
      nImages: total,
      bestK: hierarchy.clusterCount,
      silhouette: hierarchy.silhouette,
      clusterSizes,
      clusteringMethod: 'hierarchical',
      distanceMetric: hierarchy.distanceMetric,
      linkageMethod: hierarchy.linkageMethod,
      normalization: hierarchy.normalization,
      pcaDimUsed: undefined,
      projectionUsedForClustering: 'full standardized responses/features (not PCA)',
      distanceToClusterRepresentative: 'correlation_distance_to_medoid',
      // These archived artifacts describe the reference-only tree, not this tree.
      resultCsv: undefined,
      linkageCsv: undefined,
      dendrogramPng: undefined,
      filePath: undefined,
    },
    jointHierarchy: hierarchy,
  };
}
