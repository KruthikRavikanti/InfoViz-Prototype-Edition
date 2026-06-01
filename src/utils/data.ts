import { csv, json } from 'd3';
import type {
  JsonScoreObject,
  LongFormDatum,
  ModelRoiColumn,
  Roi,
  VisualizationData,
  WideCsvRow,
} from '../types/data';
import { loadClusteringData } from './clustering';

const DEFAULT_ROI_FALLBACK: Roi[] = ['ffa', 'ppa', 'eba'];

export const CSV_PATH = '/data/combined_image_means.csv';
export const SCORES_PATH = '/data/murty185_zero_index.json';
export const IMAGE_BASE_PATH = '/images';

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function parseModelRoiColumn(columnName: string, knownRois?: Iterable<Roi>): ModelRoiColumn | null {
  if (columnName === 'image_name') {
    return null;
  }

  const roiCandidates = Array.from(knownRois ?? DEFAULT_ROI_FALLBACK)
    .filter((roi) => roi !== 'Overall')
    .sort((a, b) => b.length - a.length);

  for (const roi of roiCandidates) {
    const suffix = `_${roi}`;
    if (columnName.endsWith(suffix)) {
      return {
        columnName,
        model: columnName.slice(0, -suffix.length),
        roi,
      };
    }
  }

  const lastUnderscore = columnName.lastIndexOf('_');
  if (lastUnderscore <= 0 || lastUnderscore === columnName.length - 1) {
    return null;
  }

  return {
    columnName,
    model: columnName.slice(0, lastUnderscore),
    roi: columnName.slice(lastUnderscore + 1),
  };
}

export function inferModelRoiColumns(rows: WideCsvRow[], knownRois?: Iterable<Roi>): ModelRoiColumn[] {
  const firstRow = rows[0];
  if (!firstRow) {
    return [];
  }

  return Object.keys(firstRow)
    .map((columnName) => parseModelRoiColumn(columnName, knownRois))
    .filter((column): column is ModelRoiColumn => column !== null);
}

export function inferModels(modelRoiColumns: ModelRoiColumn[]): string[] {
  return Array.from(new Set(modelRoiColumns.map(({ model }) => model))).sort((a, b) => a.localeCompare(b));
}

export function inferRois(modelRoiColumns: ModelRoiColumn[], scores?: JsonScoreObject): Roi[] {
  const scoreRois = scores ? Object.keys(scores) : [];
  const csvRois = modelRoiColumns.map(({ roi }) => roi);
  const rois = Array.from(new Set([...csvRois, ...scoreRois])).filter((roi) => roi !== 'Overall');

  return rois.sort((a, b) => {
    const preferredOrder = ['ffa', 'ppa', 'eba'];
    const aIndex = preferredOrder.indexOf(a);
    const bIndex = preferredOrder.indexOf(b);

    if (aIndex !== -1 || bIndex !== -1) {
      return (aIndex === -1 ? Number.POSITIVE_INFINITY : aIndex) - (bIndex === -1 ? Number.POSITIVE_INFINITY : bIndex);
    }

    return a.localeCompare(b);
  });
}

export function toLongForm(rows: WideCsvRow[], modelRoiColumns: ModelRoiColumn[]): LongFormDatum[] {
  return rows.flatMap((row) =>
    modelRoiColumns.flatMap((column) => {
      const rawValue = row[column.columnName];
      const value = typeof rawValue === 'number' ? rawValue : Number(rawValue);

      if (!isFiniteNumber(value)) {
        return [];
      }

      return {
        imageName: row.image_name,
        imageUrl: `${IMAGE_BASE_PATH}/${row.image_name}`,
        model: column.model,
        roi: column.roi,
        value,
        columnName: column.columnName,
      };
    }),
  );
}

export async function loadCsvRows(path = CSV_PATH): Promise<WideCsvRow[]> {
  return csv(path, (row) => {
    const parsed: WideCsvRow = {
      image_name: String(row.image_name ?? ''),
    };

    for (const [key, value] of Object.entries(row)) {
      if (key === 'image_name') {
        continue;
      }

      const numericValue = Number(value);
      parsed[key] = Number.isFinite(numericValue) ? numericValue : value;
    }

    return parsed;
  });
}

export async function loadScores(path = SCORES_PATH): Promise<JsonScoreObject> {
  const result = await json<JsonScoreObject>(path);

  if (!result || typeof result !== 'object') {
    throw new Error('Score JSON did not contain an object.');
  }

  return result;
}

export async function loadVisualizationData(): Promise<VisualizationData> {
  const [rows, scores, clustering] = await Promise.all([loadCsvRows(), loadScores(), loadClusteringData()]);
  const modelRoiColumns = inferModelRoiColumns(rows, Object.keys(scores));

  return {
    rows,
    scores,
    modelRoiColumns,
    clustering,
  };
}
