# Cortex Visualization Prototype

This repository contains the current React/Vite prototype for comparing brain-region alignment scores across computer vision models and then drilling into image-level evidence for selected ROI x model pairs.

## What The App Does

- Shows an aggregate heatmap of ROI x model scores.
- Lets users search and sort models.
- Opens a details panel for a selected cell.
- Ranks images by the matching ROI/model evidence column from the CSV.
- Supports compare mode for two evidence-backed cells.
- Includes a separate user study tab with an embedded form.

## Data In This Repo

The app loads data from `public/data/`:

- `murty185_zero_index.json`
  - aggregate ROI x model score matrix
- `combined_image_means.csv`
  - image-level evidence table used to rank stimuli for selected cells
- `voxel_clusters/` and `voxel_clusters_hierarchical_corr/`
  - selected model/ROI image clusters
- `visual_clusters/` and `visual_clusters_hierarchical_corr/`
  - hand-built visual feature clusters for the 185 images
- `dreamsim_clusters/` and `dreamsim_clusters_hierarchical_corr/`
  - DreamSim embedding clusters for the 185 images
- `ground_truth_clusters/` and `ground_truth_clusters_hierarchical_corr/`
  - patient/ROI ground-truth fMRI clusters

The app also serves image assets from `public/images/`.

The source DreamSim embedding bundle is stored in `data/dreamsim/dreamsim_embeddings.pth`.
It is not loaded by the browser directly; it is used by `scripts/data/cluster_runner.py` to generate the public DreamSim cluster CSV/JSON/PNG files.

## Data Scripts

Inspect the DreamSim bundle:

```bash
python scripts/data/dreamsim_pipeline.py --keys murty185
```

Regenerate DreamSim KMeans/euclidean clusters:

```bash
python scripts/data/cluster_runner.py dreamsim \
  --pth data/dreamsim/dreamsim_embeddings.pth \
  --embedding_key murty185 \
  --image_dir public/images \
  --out_dir public/data/dreamsim_clusters
```

Regenerate DreamSim hierarchical/correlation clusters:

```bash
python scripts/data/cluster_runner.py dreamsim-hierarchical \
  --pth data/dreamsim/dreamsim_embeddings.pth \
  --embedding_key murty185 \
  --image_dir public/images \
  --out_dir public/data/dreamsim_clusters_hierarchical_corr
```

## Current Evidence Coverage

The evidence pipeline is real, but partial.

- Aggregate score matrix coverage: `127` models across `ppa`, `ffa`, `eba`, and `Overall`
- Image-level evidence coverage: `26` models across `ppa`, `ffa`, and `eba`
- Image rows: `185`

This means the heatmap can show more models than the evidence panel can explain. In compare mode, the UI intentionally restricts selection to cells that have matching CSV evidence.

## Tech Stack

- React
- TypeScript
- Vite
- D3 utilities for loading and processing data

## Project Structure

- `src/App.tsx`
  - top-level app state and data loading
- `src/components/`
  - heatmap, controls, details panel, compare drawer, and user study UI
- `src/utils/data.ts`
  - CSV/JSON loading and model/ROI parsing
- `src/utils/evidence.ts`
  - image ranking and evidence summaries
- `src/utils/compare.ts`
  - overlap and comparison summaries
- `public/data/`
  - score matrix and image-level evidence CSV
- `public/images/`
  - evidence image assets

## Local Development

Follow these steps to set up and run the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/audreyzhao75/InfoViz-Prototype.git
   cd InfoViz-Prototype
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   This will start the app locally at http://localhost:5174. Port 5174 is fixed so Virtual Cortex can use port 5173 at the same time.

4. **Build for production:**
   ```bash
   npm run build
   ```

5. **Preview the production build:**
   ```bash
   npm run preview
   ```

## Cleanup Notes

This repo previously tracked generated artifacts and dependency folders. The `.gitignore` now excludes:

- `node_modules/`
- `dist/`
- `.DS_Store`
- TypeScript build info files

If you want the Git history to stop tracking those files, remove them from the index in a cleanup commit.

## Cortex comparison-session mode

The standalone visualization remains the default. When Virtual Cortex opens a URL such as
`http://localhost:5174/#session=<opaque-token>`, InfoViz retrieves the temporary comparison from Cortex Core,
validates it against the local reference manifest, removes the token from browser history after a successful load,
and keeps the session only in memory.

Copy `.env.example` to `.env` when the services use non-default origins:

```bash
VITE_CORTEX_CORE_URL=http://localhost:5556
VITE_CORTEX_APP_URL=http://localhost:5173
```

The dev server listens on the university machine network interface as well as localhost. For remote development,
replace `localhost` with the machine hostname in both values and include that InfoViz origin in Core's allowed origins.

The session view always labels the standardized comparison separately from the user's original Cortex request:
comparison evidence uses NSD-1000 across FFA, PPA, and EBA. Uploaded images are predictions only and are never
inserted into measured patient/fMRI clusters. Voxel uploads use saved PCA/K-means transforms; static t-SNE remains
reference-only. DreamSim upload placement is currently disabled because the archived 1,792-feature reference bundle
is incompatible with the current 768-feature inference path.
