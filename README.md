# Cortex Model & Stimulus Explorer

This React/Vite visualization compares model alignment across brain regions and explores image responses for selected model–region pairs.

## What The App Does

- Shows an aggregate heatmap of ROI x model scores.
- Lets users search and sort models.
- Opens a details panel for a selected cell.
- Ranks images by the matching ROI/model evidence column from the CSV.
- Supports compare mode for two evidence-backed cells.
- Keeps the four image-cluster charts intact and groups the anatomical Brain Map and non-anatomical Voxel Selectivity Explorer under Brain & voxels.
- Shows uploaded-stimulus and Murty-185 reference predictions on the NSD Subject 1 surface in the 3D Brain Map.
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
- `voxel_selectivity_atlas/`
  - lazy-loaded, model-specific Subject 1 voxel-selectivity artifacts derived from the 185 reference predictions

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

Regenerate the Subject 1 artifacts for the Voxel Selectivity Explorer:

```bash
python scripts/data/build_selectivity_atlas_artifacts.py
```

This artifact builder follows the reviewed mentor pipeline exactly for clustering: per-voxel z-scoring across the
185 images, PCA to 35 dimensions, L2 normalization, and Euclidean KMeans at fixed `k=14`. It uses only the audited
Subject 1 prefixes (156 FFA, 518 EBA, and 170 PPA voxels) and fits them jointly so cluster labels have one shared
meaning across ROIs. The 3D cosine-UMAP coordinates are display-only and are not anatomical positions.

The supplied mentor script states that `k=14` was split-half stable, but it does not run that analysis. The browser
therefore describes the 14 clusters as exploratory and keeps the analysis and six freshly computed split-half
adjusted Rand indices in a collapsible **Method & quality** section. These diagnostics do not establish `k=14`
as an optimal or biological parcellation. The source values are
model predictions for the category-imbalanced Murty-185 stimulus set, not measured Subject 1 activity.

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

## Visual Identity

The visualization follows the sibling Virtual Cortex Lab: warm ivory surfaces, aubergine accents,
Source Serif 4 headings, Inter controls, and IBM Plex Mono labels. Shared theme tokens live in
`src/styles/global.css`; chart palettes remain independent so scientific color encodings keep their meaning.
The Cortex masthead links home and back to the Lab, using `VITE_CORTEX_APP_URL` when configured.

Model search, ranking, score labels, and two-selection comparison are grouped in the model browser.
The selected model–region workspace has **Image responses**, **Image clusters**, and **Brain & voxels**
tabs; JSON export and metadata copying remain available across all three. The heatmap uses the quantitative
Cortex scoreboard's cream–lavender–purple palette with compact square cells. Its full-data color scale stays
fixed when searching; score values are unchanged and are not reinterpreted as ceiling-normalized scores.

**Brain & voxels** contains the cortical 3D Brain Map and the non-anatomical Voxel Selectivity Explorer.
The brain image picker is searchable and bounded regardless of upload count, with previous/next navigation
and a selected-image preview. Display controls collapse on narrow screens so the cortex stays visible.
Reference-image selection is local to the brain map; selecting an upload still updates the shared upload focus.

Hovering a mapped vertex preserves its voxel details and previews the image with the highest raw predicted
response at its dominant source voxel. Rankings use only the active **Uploads** or **Murty-185** collection;
picker search does not change that scope, and hovering does not select another brain image. The signed maximum
is used (the least-negative prediction if all are negative), with exact ties flagged and the first image shown.
This is an in-collection prediction maximum, not measured activation or a maximum of the blended surface value.

The 3D Brain Map supports all 185 reference images for the 26 audited NSD-1000 models, even without an upload
session. `scripts/data/build_brain_reference_artifacts.py` exports the exact raw float32 Subject 1 prefixes
(156 FFA, 170 PPA, 518 EBA) from the existing prediction bundle; it does not run inference or use normalized
clustering coordinates. Canonical image identities, source hashes, voxel ordering, model coverage, and response
space are checked before export. The lazy-loaded binary assets and their index are checksum-verified in the
browser. Uploads and references share a fixed scale only for the same model/reference bundle. These are model
predictions, not measured BOLD responses; the existing three unmapped EBA responses remain unplotted.
Unsupported models are explicitly unavailable rather than receiving substitute predictions. The selectivity
atlas remains reference-only and is not refitted by uploads.

The selectivity orientation widget labels the saved coordinates as **UMAP 1, 2, and 3** and follows the camera.
These are arbitrary similarity-space axes, not left/right, anterior/posterior, or superior/inferior directions.
ROI visibility changes and resizing preserve the view; **Reset view** restores the full model overview.
Focus the plot to rotate with arrow keys, zoom with +/−, inspect individual visible voxels with [ / ] or
Page Up/Down (Home/End jump to the endpoints), and select their cluster with Enter/Space. Mouse and keyboard
inspection show the same voxel/cluster/PCA-distance details, with tooltips kept inside the plot.
The selectivity manifest digest is pinned in `src/services/selectivityAtlasAssets.ts` before either model
availability or atlas data are trusted. An intentional artifact regeneration requires reviewing the bundle
and updating that pin; child summary and point files remain independently checksum-verified.

Regenerate the separate reference-brain assets with the audited private source bundle present:

```bash
python scripts/data/build_brain_reference_artifacts.py
```

The generated manifest SHA-256 is pinned in `src/services/brainReferenceAssets.ts`; an intentional source-bundle
update requires reviewing and updating that pin. Neither existing anatomy assets nor comparison artifacts are rewritten.

The four-chart layout distinguishes **Predicted-response clusters**, **Visual-feature clusters**,
**DreamSim similarity clusters**, and **Measured fMRI clusters**. Their dots are images; dots in the
Voxel Selectivity Explorer are voxels. Model alignment scores and mean predicted voxel responses
are labeled separately. Fit metrics, source columns, mapping provenance, and extra comparison
galleries remain available in expandable details rather than crowding the main views.

In hierarchical mode, new upload sessions jointly cluster the 185 reference images and all uploads in the
predicted-response and compatible visual-feature charts. Core uses the full reference-standardized vectors,
pairwise correlation distance, average linkage, and the best silhouette among 2–5 cluster cuts. Cluster labels,
combined sizes, representative previews, and the interactive dendrogram all use this session-specific fit.
Uploads are actual tree leaves; medoids summarize the resulting clusters rather than assigning membership.
Reference normalization and PCA display positions stay fixed. Upload coordinates are aligned only after
all reference image IDs and coordinates pass a sign-only check; unverified axes suppress scatter markers
without hiding the joint tree. The fit is computed once per new upload/model session and persisted, not
recalculated on hover or refresh. Cluster labels and reference memberships may change between datasets;
these exploratory predicted-response groups are not validated neural categories.

Legacy sessions without a joint tree show reference-only hierarchical plots and instructions to reopen
Explore visualization in Virtual Cortex. They never silently fall back to nearest-medoid assignments.
k-means remains a fixed-reference overlay. DreamSim and measured-fMRI upload safeguards remain unchanged.

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
`http://localhost:5173/infoviz/#session=<opaque-token>`, InfoViz retrieves the temporary comparison from Cortex Core,
validates it against the local reference manifest, removes the token from browser history after a successful load,
and keeps the session only in memory.

The sibling-repository development defaults are already equivalent to:

```bash
VITE_CORTEX_CORE_URL=/cortex-core
CORTEX_CORE_PROXY_TARGET=http://127.0.0.1:5556
VITE_INFOVIZ_BASE_PATH=/infoviz/
VITE_CORTEX_APP_URL=http://localhost:5173
```

In the three-service development setup, Cortex App proxies `/infoviz/` to this
server. Remote SSH users therefore forward only Cortex App's port 5173; the
browser never connects directly to ports 5174 or 5556. A direct visit to
`http://localhost:5174` redirects to `/infoviz/` and remains standalone static
mode. Set `VITE_INFOVIZ_BASE_PATH=/` only when a deployment must live at the
origin root.

The session view always labels the standardized comparison separately from the user's original Cortex request:
comparison evidence uses NSD-1000 across FFA, PPA, and EBA. Uploaded images are predictions only and are never
inserted into measured patient/fMRI clusters. Voxel uploads use saved PCA/K-means transforms; static t-SNE remains
reference-only. DreamSim upload placement is currently disabled because the archived 1,792-feature reference bundle
is incompatible with the current 768-feature inference path.
