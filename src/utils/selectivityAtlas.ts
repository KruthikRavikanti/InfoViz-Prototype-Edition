import type { SelectivityAtlasRoi } from '../types/selectivityAtlas';
import { publicPath } from './publicPath';

export const SELECTIVITY_CLUSTER_COLORS = [
  '#4C72B0',
  '#DD8452',
  '#55A868',
  '#C44E52',
  '#8172B3',
  '#937860',
  '#DA8BC3',
  '#CCB974',
  '#64B5CD',
  '#E7969C',
  '#686868',
  '#17A9B8',
  '#A7A918',
  '#1F4E79',
] as const;

export const SELECTIVITY_ROI_LABELS: Record<SelectivityAtlasRoi, string> = {
  ffa: 'Fusiform Face Area',
  eba: 'Extrastriate Body Area',
  ppa: 'Parahippocampal Place Area',
};

export function selectivityImageUrl(imageId: string): string {
  return publicPath(`/images/${imageId}`);
}
