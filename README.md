# Cortex-Compass

Cortex-Compass is Virtual Cortex's model and stimulus visualization. It compares model alignment across brain regions and explores image responses for selected model–region pairs.

Open it directly or choose **Open Cortex-Compass** after generating predictions in the Virtual Cortex Lab. Virtual Cortex's `/cortex-compass/` workspace opens in a separate tab, keeps its navigation visible, and leaves the original Lab results available. The existing `/infoviz/` route, repository name, and comparison-session protocol are unchanged.

## What The App Does

- Shows an aggregate heatmap of ROI x model scores.
- Lets users search and sort models.
- Opens a details panel for a selected cell.
- Ranks images by the matching ROI/model evidence column from the CSV.
- Supports compare mode for two evidence-backed cells.
- Keeps the four image-cluster charts intact and groups the Brain Map and Voxel Selectivity Explorer under Brain & selectivity.
- Shows uploaded-stimulus and Murty-185 reference predictions on the NSD Subject 1 surface in the 3D Brain Map.

## Data In This Repo

The app loads prediction and analysis data from `public/data/`. Model alignment uses the
Virtual Cortex quantitative scoreboard's **NSD-1000 training → Murty185 evaluation →
Univariate (Pearson r)** scores, not its broader multi-dataset Global Score:

- `/assets/data/new/standardized_results_nsd_1000_models_univariate.json`
  - authoritative score source when embedded in Virtual Cortex; uses each
    `[roi][model].murty185` pair exactly as the scoreboard does
- `scoreboard/standardized_results_nsd_1000_models_univariate.json`
  - matching standalone snapshot; `scoreboard/source.json` records source repository,
    commit, file checksum, training/evaluation datasets, and metric
- `murty185_zero_index.json`
  - retained historical flattened score matrix for reference-generation and release audits;
    no longer the browser's alignment source
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

Check the standalone score snapshot against the Cortex App checkout:

```bash
node scripts/data/sync_scoreboard_scores.mjs --check
```

To refresh the snapshot and its provenance, review and apply the generated patch:

```bash
node scripts/data/sync_scoreboard_scores.mjs --patch | apply_patch
perl -pi -e 'chomp if eof' public/data/scoreboard/standardized_results_nsd_1000_models_univariate.json
node scripts/data/sync_scoreboard_scores.mjs --check
```

The script detects the containing Cortex App checkout when installed at
`cortex-app/apps/cortex-compass/`; an independent checkout uses sibling
`../cortex-app`. Use `--source-root /path/to/cortex-app` to select another checkout.
The EOF formatting command keeps the snapshot byte-identical to the source JSON, which has
no terminal newline. The embedded view reads the host's score file directly and does not silently fall back to
an older snapshot when that source is unavailable. This keeps the table and scoreboard
on the same release. Score updates do not retrain models or regenerate prediction artifacts.

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

Inspect the Subject 1 Voxel Selectivity Explorer builder's explicit staging options:

```bash
python scripts/data/build_selectivity_atlas_artifacts.py --help
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

The `126` models in the audited reference bundle have image-response evidence for all
`185` Murty reference images across FFA, PPA, and EBA. This includes response rankings,
both predicted-response clustering methods, Subject 1 brain maps, and voxel-selectivity
artifacts. The updated scoreboard also includes the score-only `pixels` baseline,
for `127` scored models plus the **noise ceiling**. Neither `pixels` nor the noise
ceiling has image-response evidence in this bundle, so both remain unavailable as
prediction evidence in Compare mode. The noise ceiling is not a runnable model.

All 126 models use their registered Cortex Core backbones and existing trained
NSD-1000 readouts, evaluated on the canonical `public/images/image_001.png` through
`image_185.png` files. No model is substituted and no readout is retrained.
The original 26-model bundle and the subsequent mixed-policy 126-model release
are retained as recoverable historical snapshots, not silently overwritten.
The active standardized release regenerates every model's predictions and refits
its derived image clusters, Subject 1 brain responses, and selectivity atlas.
Numerical values, near-tied image ranks, and cluster partitions can therefore
change; the processing methods and parameters do not. Standardized reference
means, image clustering, and voxel-selectivity analyses start from the exact
float32 prediction values, promoted to float64 for their existing analyses.
The brain map retains those same audited float32 Subject 1 values. This avoids
a second, slightly different interpretation of decimal CSV values, especially
for nearly constant random-control responses. The full pooled readouts contain 716 FFA,
753 PPA, and 2,045 EBA columns, ordered by NSD subjects 1, 2, 5, and 7; anatomical
views use only the audited Subject 1 prefixes.

All 126 models use one explicit full-FP32 policy: TF32 and autocasting are
disabled, with deterministic cuDNN and highest-precision float32 matrix multiplication.
The same policy is used for reference generation and live upload inference, avoiding
batch-dependent TF32 rounding differences. This controls numerical arithmetic;
it does not establish better biological predictive accuracy. Core serializes precision-sensitive inference
and restores settings on exit so concurrent requests cannot mix these policies.
The scoreboard retains its historical benchmark scores; those scores are not
new evaluations under the standardized execution policy.

VOne-AlexNet and VOne-CORnet-S retain their configured neuronal noise. Their saved
reference predictions are seeded samples, not averages, and the UI warns that repeated
predictions may vary. The registered VOne-RN50 checkpoint has no active neuronal noise
and is not labeled stochastic. Random-backbone variants remain distinct models and
use the initialization recorded in their inference provenance.
The random AlexNet and VGG16/VGG19 constructors set their seed before allocating weights,
so the first ROI loaded uses the same recorded initialization as subsequent ROIs.
The original 26-model archives do not record backbone-state hashes, so changes
in those archives cannot all be attributed solely to FP32 arithmetic. Nearly
flat random-control responses can also reorder substantially despite small
absolute changes. The release comparison records both response and rank changes.
For AlexNetRandom specifically, the historical checkpoints contain readout
weights only, and neither the cached encodings nor the backbone cache records
the training encoder's initialization. The controlled seed-0 TF32 test does
not explain the archived response differences. Its current reference/live
consistency is verified, but its historical training-encoder pairing remains
unverified; interpret this random control accordingly.

Coverage changes have a new reference version. Reopen Cortex-Compass from the Lab
to create a fresh comparison after updating the bundle; old-version session tokens
are not silently mixed with the expanded reference data.

### Maintaining coverage

`scripts/data/generate_model_reference_predictions.py` generates prediction triads in
an ignored staging directory and records image, backbone, readout, preprocessing,
seed, numerical precision, helper-source hash, and noise-policy provenance. Run GPU inference with `CUDA_VISIBLE_DEVICES=0`
in this shared local environment. Its default is offline; explicit `--allow-network`
is needed for an uncached official checkpoint. The local backbone overlay uses
readable shared weights without writing into the shared research directories.

For a uniform-policy release, `--all-scored` regenerates all 126 models into a
fresh `artifacts/precision-standardization/raw-fp32` directory. It does not reuse
old provenance with a new helper hash. Freeze the inference sources before
generation; every provenance record binds the observed numerical policy and
actual source bytes. Changing those sources requires a new verified run.

Full regeneration is a separate operation from append-only expansion:
`refresh_reference_coverage.py` requires the exact reviewed 126-model parent
snapshot, refits every model with the existing algorithms and parameters, and
records `prediction_standardization` lineage. The brain and atlas builders use
`--standardize-from-baseline` to export/refit all models from that new release.
Its `analysis_input_policy` explicitly binds float32-to-float64 ingestion;
the legacy builders' decimal-CSV ingestion remains unchanged.
`verify_reference_refresh.py` checks the complete parent snapshot, all new raw
sources and inference proofs, fitted transforms, plotted cluster identities,
and the independent spatial audit. `compare_reference_releases.py` measures
changes in raw responses, saved ranking means, image midranks, and image/voxel cluster partitions;
it distinguishes changes to displayed means from changes to raw float32 voxel averages.
adjusted Rand indices avoid mistaking cluster-label permutations for changes.

Only after staged verification should identity pins be reviewed and
`install_reference_refresh.py --servers-stopped` used. Its scoped transaction
retains the entire prior release and all historical raw CSVs, adds the fresh
sources without collisions, and restores the previous state on failure.
`verify_live_coverage.py --all-supported` checks the real HTTP comparison flow
for every published model; noisy VOne models retain explicit stochastic checks
instead of false cross-draw equality assertions. The historical scoreboard,
visual-feature, DreamSim, measured-fMRI, and anatomy assets remain unchanged.

`extend_reference_coverage.py` requires an explicit complete target roster and
fits only the added models. Use the spatial builders' `--append-baseline-brain`
and `--append-baseline-atlas` modes with separate, fresh output directories.
`verify_coverage_release.py` checks the whole staged release against the saved
baseline, raw predictions, transforms, model identities, and spatial checksums.
For the corrected FP32 release, use `artifacts/coverage-expansion/raw-fp32`
as the raw-input path: pass it via `--new-raw-dir` to the extension builder,
and via `--raw-root` to both the release verifier and installer.
Keep the immutable original `artifacts/coverage-expansion/baseline` as the
append baseline; use fresh output directories for a new candidate.
Only after review should the frontend coverage pins and Core spatial-audit pin
be updated and `install_coverage_release.py --servers-stopped` used during local
maintenance. The installer retains the original directories in a version-specific
backup and restores them if installation fails. A same-roster correction also
requires an explicit byte-verified snapshot of the installed candidate; its prior
added raw sources are moved into the retained backup, never overwritten or deleted.
`verify_live_coverage.py` then
tests real upload sessions, rankings, joint clustering, previews, and Subject 1
responses against the published predictions.

Legacy builders retain their original strict 26-model contract; they are not a
shortcut for regenerating the expanded release. Consult each script's `--help`
for explicit staging paths. Cortex Core must use the prepared overlay via
`CORTEX_BACKBONE_DIR`; its scoped cache helper isolates constructor-specific
Torch Hub overrides so loading DreamSim cannot redirect later models' caches.

### Moving the standardized release to another machine

Git contains the public 126-model views and their reviewed identity pins.
The private Core bundle in `artifacts/reference` is intentionally ignored:
a fresh clone alone cannot create matching upload comparisons. Transfer that
verified directory separately, use the matching Cortex Core precision/cache/audit
changes and trained readout resources, and set `CORTEX_REFERENCE_ARTIFACT_DIR`
to its absolute path. The installed release version is
`3345e5fb332f4046c0b3a76a35f24a027dd0b6de0f62bb5e97eecda62c06b67d`;
its private `manifest.json` SHA-256 is
`74c282968b2d2d807f499e933f68d64668650ce8f2a783d4fb2931e0ab9b77f2`.
Retain the immutable baseline and FP32 provenance under
`artifacts/precision-standardization`, together with the earlier
`artifacts/coverage-expansion` snapshots, as release audit material. The local
`pre-install-3345e5fb332f4046c0b3a76a35f24a027dd0b6de0f62bb5e97eecda62c06b67d`
backup retains the previous mixed-policy release, and all historical raw CSVs
remain in place. Do not run the legacy
26-model builder over the expanded inputs to try to reproduce this release.

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
Inside Virtual Cortex's same-origin full-page workspace, `?embedded=1` hides only
this duplicate masthead; the title, reference/model context, charts, and dialogs
remain. The flag has no effect outside a frame, so standalone navigation stays intact.

Below the Cortex-Compass title, a compact context summary identifies the Murty-185 reference images,
the selected model and region, and NSD-1000 predictions where image-response data are available.
Lab sessions separately show the user's original model, mapping dataset, and ROI. Browsing another
model or comparing reference-only cells does not relabel the uploaded predictions; score-only models
are explicitly distinguished from models with image-response data.

Model search, ranking, score labels, and two-selection comparison are grouped in the model browser.
The selected model–region workspace has **Predicted responses**, **Image similarity**, and **Brain & selectivity**
tabs; JSON export and metadata copying remain available across all three. The heatmap uses the quantitative
Cortex scoreboard's cream–lavender–purple palette with compact square cells. Numbers, comparisons,
and ranking use the **raw Pearson correlations**. Colors use the scoreboard's **stored
ceiling-normalized values**, clipped to 0–1, with the same fixed legend; filtering does not
rescale them. Overall alignment is the arithmetic mean of raw FFA, PPA, and EBA scores for
this evaluation set. The noise ceiling is kept above the model rows and excluded from model
ranks and the model count (127 models plus the noise ceiling). These ranks match the scoreboard
with the same training, evaluation, and metric selected—not its multi-dataset default ranking.

**Predicted responses** covers response rankings, distributions, and upload-to-reference comparisons.
**Image similarity** groups images by predicted-response patterns, visual features, or measured fMRI.
**Brain & selectivity** contains the cortical 3D Brain Map and the Voxel Selectivity Explorer,
which offers cortical and non-anatomical similarity views.
The brain image picker is searchable and bounded regardless of upload count, with previous/next navigation
and a selected-image preview. Display controls collapse on narrow screens so the cortex stays visible.
Reference-image selection is local to the brain map; selecting an upload still updates the shared upload focus.

Hovering a mapped vertex preserves its voxel details and previews the image with the highest raw predicted
response at its dominant source voxel. Rankings use only the active **Uploads** or **Murty-185** collection;
picker search does not change that scope, and hovering does not select another brain image. The signed maximum
is used (the least-negative prediction if all are negative), with exact ties flagged and the first image shown.
This is an in-collection prediction maximum, not measured activation or a maximum of the blended surface value.

The 3D Brain Map supports all 185 reference images for the 126 audited NSD-1000 models, even without an upload
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
The selectivity manifest digest is pinned in `src/generated/referenceCoverage.ts` before either model
availability or atlas data are trusted. An intentional artifact regeneration requires reviewing the bundle
and updating that pin; child summary and point files remain independently checksum-verified.

Inspect the separate reference-brain builder's staging options (an expanded bundle
requires append-baseline mode and reviewed new-model provenance):

```bash
python scripts/data/build_brain_reference_artifacts.py --help
```

The generated manifest SHA-256 and exact model roster are pinned in `src/generated/referenceCoverage.ts`; an intentional source-bundle
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

Legacy sessions without a joint tree show reference-only hierarchical plots and instructions to choose
**Open Cortex-Compass** again in Virtual Cortex. They never silently fall back to nearest-medoid assignments.
k-means remains a fixed-reference overlay. DreamSim and measured-fMRI upload safeguards remain unchanged.

## Project Structure

- `src/App.tsx`
  - top-level app state and data loading
- `src/components/`
  - heatmap, controls, details panel, comparison views, and brain/voxel explorers
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

Cortex-Compass remains an independent repository. It can be checked out standalone
or included in Cortex App as the Git submodule `apps/cortex-compass/`.

### Inside Cortex App

Clone Cortex App with `--recurse-submodules`, or initialize the linked source from
an existing checkout. Run the following from the Cortex App root:

```bash
git submodule update --init --recursive
npm ci
npm --prefix apps/cortex-compass ci
npm run build
```

The parent build builds Cortex-Compass and includes its generated files at
`dist/infoviz/`, alongside Cortex App's pages. The deployed visualization does not
need a separate frontend server. Local development still uses two frontend
processes: run `npm run dev` and `npm run dev:compass` from the Cortex App root.

The parent repository's Git link pins an exact Cortex-Compass commit, not the
latest commit on `IUI`. To release a child change, commit and push it in this
repository, then explicitly review and update the parent's pinned commit. Normal
clones and builds do not automatically advance that pointer.

Developers and deployment checkouts need read access to both repositories. If
Cortex-Compass is private, its submodule checkout must authenticate too; credentials
must not be committed. Private reference artifacts and trained models remain on
the separately configured Cortex Core backend, not inside the submodule.

### Standalone

Follow these steps to run the independent checkout:

1. **Clone the repository:**
   ```bash
   git clone --branch IUI https://github.com/KruthikRavikanti/InfoViz-Prototype-Edition.git
   cd InfoViz-Prototype-Edition
   ```

2. **Install dependencies:**
   ```bash
   npm ci
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

## Cortex-Compass comparison sessions

Standalone Cortex-Compass remains the default. When Virtual Cortex opens a URL such as
`http://localhost:5173/infoviz/#session=<opaque-token>`, Cortex-Compass retrieves the temporary comparison from Cortex Core,
validates it against the local reference manifest, removes the token from browser history after a successful load,
and keeps the session only in memory.

The integrated host opens `/cortex-compass/` and passes that token to the embedded
page's fragment. The host retains the opaque token in the current tab's history
state to support refresh; it does not persist comparison data in browser storage.
The frame sends only loading/ready/error and return-to-Lab messages to the exact
same-origin parent. The host validates the message origin and frame identity.

The source remains separately versioned even when checked out inside
`cortex-app/apps/cortex-compass/`. Cortex App's combined build serves this repo's
compiled output at `/infoviz/` alongside the host's pages. Upload comparisons still
require the matching Core backend and private reference bundle; linking and
building the frontend does not deploy those resources, and the local Vite proxy
is not a production backend configuration.

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

New Lab sessions prepare the selected model plus `efficient_net`, `clip_rn50`, and
`vgg16_imagenet1kv1`, without duplicates (three or four models total). The selected
model remains the initial view. Buttons in the comparison banner switch the
same stimuli between these models while preserving the selected ROI and image.
Each model has its own predicted means, ranks, response-pattern clusters, and
audited brain vectors. Images and model-independent visual features are shared;
missing brain vectors are never replaced with another model's responses.
The Lab prepares one ordinary single-model comparison session per model, using
the existing `/comparison-sessions` API without `additional_models`. Its fragment
handoff is `multi-v1.<primary-token>.<companion-token>...` (three or four unique
tokens); this is browser coordination, not a new backend endpoint. Cortex-Compass
loads each session, verifies matching Lab settings, reference release, image order,
and the SHA-256 of the sanitized temporary PNG images, then maps each session's
independent upload IDs to the primary image IDs. Image verification uses at most
four concurrent downloads and requires HTTPS or a secure localhost connection.
The internal `modelResults` view state is unchanged, all models retain their own
responses and brain vectors, and the combined expiration is the earliest session
expiration. Model-independent visual and DreamSim evidence stays with the primary
session. A mismatched, missing, or expired companion fails the whole handoff rather
than displaying a partial comparison. Legacy single-token sessions, including old
combined `model_results` payloads, continue to work. Other heatmap models and the
two-cell Compare view remain reference-only for those uploads.
