import type {
  ClusterComparisonSummary,
  ClusterIntersection,
  ClusterPoint,
  CompareSummary,
  EvidenceImage,
  EvidenceView,
  ImageSetComparison,
  RankSimilaritySummary,
  SelectedHeatmapCell,
} from '../types/data';

function imageNameSet(images: EvidenceImage[]): Set<string> {
  return new Set(images.map((image) => image.imageName));
}

export function compareImageSets(aImages: EvidenceImage[], bImages: EvidenceImage[]): ImageSetComparison {
  const bNames = imageNameSet(bImages);
  const aNames = imageNameSet(aImages);

  return {
    overlap: aImages.filter((image) => bNames.has(image.imageName)),
    uniqueA: aImages.filter((image) => !bNames.has(image.imageName)),
    uniqueB: bImages.filter((image) => !aNames.has(image.imageName)),
  };
}

function difference(aValue: number | null, bValue: number | null): number | null {
  if (aValue === null || bValue === null) {
    return null;
  }

  return aValue - bValue;
}

function pearsonCorrelation(aValues: number[], bValues: number[]): number | null {
  if (aValues.length !== bValues.length || aValues.length < 2) {
    return null;
  }

  const aMean = aValues.reduce((sum, value) => sum + value, 0) / aValues.length;
  const bMean = bValues.reduce((sum, value) => sum + value, 0) / bValues.length;
  let numerator = 0;
  let aSumSquares = 0;
  let bSumSquares = 0;

  for (let i = 0; i < aValues.length; i += 1) {
    const aCentered = aValues[i] - aMean;
    const bCentered = bValues[i] - bMean;
    numerator += aCentered * bCentered;
    aSumSquares += aCentered ** 2;
    bSumSquares += bCentered ** 2;
  }

  const denominator = Math.sqrt(aSumSquares * bSumSquares);
  if (denominator === 0) {
    return null;
  }

  const correlation = numerator / denominator;
  return Math.max(-1, Math.min(1, correlation));
}

export function calculateRankSimilarity(aImages: EvidenceImage[], bImages: EvidenceImage[]): RankSimilaritySummary {
  const bRankByImage = new Map(bImages.map((image) => [image.imageName, image.rank]));
  const aRanks: number[] = [];
  const bRanks: number[] = [];

  for (const image of aImages) {
    const bRank = bRankByImage.get(image.imageName);
    if (bRank === undefined) {
      continue;
    }

    aRanks.push(image.rank);
    bRanks.push(bRank);
  }

  const spearmanRho = pearsonCorrelation(aRanks, bRanks);

  return {
    spearmanRho,
    similarityScore: spearmanRho === null ? null : ((spearmanRho + 1) / 2) * 100,
    sharedImageCount: aRanks.length,
  };
}

function choose2(value: number): number {
  return value < 2 ? 0 : (value * (value - 1)) / 2;
}

function adjustedRandIndex(intersections: ClusterIntersection[], aCounts: Map<string, number>, bCounts: Map<string, number>, n: number): number | null {
  if (n < 2) {
    return null;
  }

  const totalPairs = choose2(n);
  const index = intersections.reduce((sum, cell) => sum + choose2(cell.count), 0);
  const aPairSum = Array.from(aCounts.values()).reduce((sum, count) => sum + choose2(count), 0);
  const bPairSum = Array.from(bCounts.values()).reduce((sum, count) => sum + choose2(count), 0);
  const expectedIndex = (aPairSum * bPairSum) / totalPairs;
  const maxIndex = (aPairSum + bPairSum) / 2;
  const denominator = maxIndex - expectedIndex;

  if (denominator === 0) {
    return index === maxIndex ? 1 : null;
  }

  return (index - expectedIndex) / denominator;
}

function normalizedMutualInformation(intersections: ClusterIntersection[], aCounts: Map<string, number>, bCounts: Map<string, number>, n: number): number | null {
  if (n < 2) {
    return null;
  }

  let mutualInformation = 0;
  for (const cell of intersections) {
    const aCount = aCounts.get(cell.aLabel) ?? 0;
    const bCount = bCounts.get(cell.bLabel) ?? 0;
    if (cell.count === 0 || aCount === 0 || bCount === 0) {
      continue;
    }

    mutualInformation += (cell.count / n) * Math.log((cell.count * n) / (aCount * bCount));
  }

  const entropy = (counts: Map<string, number>) =>
    Array.from(counts.values()).reduce((sum, count) => {
      const probability = count / n;
      return probability === 0 ? sum : sum - probability * Math.log(probability);
    }, 0);

  const aEntropy = entropy(aCounts);
  const bEntropy = entropy(bCounts);
  const denominator = Math.sqrt(aEntropy * bEntropy);
  return denominator === 0 ? null : mutualInformation / denominator;
}

function pairwiseAgreement(aLabels: string[], bLabels: string[]): { pairAgreement: number | null; coClusterJaccard: number | null } {
  if (aLabels.length !== bLabels.length || aLabels.length < 2) {
    return { pairAgreement: null, coClusterJaccard: null };
  }

  let matchingDecisions = 0;
  let togetherInBoth = 0;
  let togetherInEither = 0;
  const totalPairs = choose2(aLabels.length);

  for (let i = 0; i < aLabels.length; i += 1) {
    for (let j = i + 1; j < aLabels.length; j += 1) {
      const sameA = aLabels[i] === aLabels[j];
      const sameB = bLabels[i] === bLabels[j];

      if (sameA === sameB) {
        matchingDecisions += 1;
      }
      if (sameA && sameB) {
        togetherInBoth += 1;
      }
      if (sameA || sameB) {
        togetherInEither += 1;
      }
    }
  }

  return {
    pairAgreement: matchingDecisions / totalPairs,
    coClusterJaccard: togetherInEither === 0 ? null : togetherInBoth / togetherInEither,
  };
}

export function calculateClusterComparison(aPoints: ClusterPoint[], bPoints: ClusterPoint[]): ClusterComparisonSummary | null {
  const bPointByImage = new Map(bPoints.map((point) => [point.imageName, point]));
  const aLabels: string[] = [];
  const bLabels: string[] = [];
  const aCounts = new Map<string, number>();
  const bCounts = new Map<string, number>();
  const intersectionCounts = new Map<string, ClusterIntersection>();

  for (const aPoint of aPoints) {
    const bPoint = bPointByImage.get(aPoint.imageName);
    if (!bPoint) {
      continue;
    }

    const aLabel = aPoint.clusterLabel;
    const bLabel = bPoint.clusterLabel;
    aLabels.push(aLabel);
    bLabels.push(bLabel);
    aCounts.set(aLabel, (aCounts.get(aLabel) ?? 0) + 1);
    bCounts.set(bLabel, (bCounts.get(bLabel) ?? 0) + 1);

    const key = `${aLabel}\u0000${bLabel}`;
    const existing = intersectionCounts.get(key);
    if (existing) {
      existing.count += 1;
    } else {
      intersectionCounts.set(key, { aLabel, bLabel, count: 1 });
    }
  }

  if (aLabels.length === 0) {
    return null;
  }

  const intersections = Array.from(intersectionCounts.values());
  const sortedIntersections = intersections.sort((a, b) => b.count - a.count);
  const pairMetrics = pairwiseAgreement(aLabels, bLabels);

  return {
    adjustedRandIndex: adjustedRandIndex(intersections, aCounts, bCounts, aLabels.length),
    normalizedMutualInformation: normalizedMutualInformation(intersections, aCounts, bCounts, aLabels.length),
    pairAgreement: pairMetrics.pairAgreement,
    coClusterJaccard: pairMetrics.coClusterJaccard,
    sharedImageCount: aLabels.length,
    clusterCountA: aCounts.size,
    clusterCountB: bCounts.size,
    intersections: sortedIntersections,
    largestIntersections: sortedIntersections.slice(0, 6),
  };
}

export function buildCompareSummary(
  aCell: SelectedHeatmapCell,
  aEvidence: EvidenceView | null,
  bCell: SelectedHeatmapCell,
  bEvidence: EvidenceView | null,
): CompareSummary | null {
  if (!aEvidence || !bEvidence) {
    return null;
  }

  return {
    top: compareImageSets(aEvidence.topImages, bEvidence.topImages),
    bottom: compareImageSets(aEvidence.bottomImages, bEvidence.bottomImages),
    aggregateScoreDifference: difference(aCell.score, bCell.score),
    imageMeanDifference: aEvidence.stats.mean - bEvidence.stats.mean,
    rankSimilarity: calculateRankSimilarity(aEvidence.images, bEvidence.images),
  };
}
