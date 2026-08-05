# DreamSim Embeddings

This folder stores the precomputed DreamSim embedding bundle used by the clustering pipeline.

## Files

- `dreamsim_embeddings.pth`
  - PyTorch dict of dataset keys to embedding tensors.
  - The app uses `emb["murty185"]`, shape `(185, 1792)`.

## Row Mapping

For this repo, the `murty185` rows are treated as aligned with:

```text
public/images/image_001.png
public/images/image_002.png
...
public/images/image_185.png
```

The clustering script validates this by checking that `public/images` contains 185 supported image files.

## Inspect The Embeddings

```bash
python scripts/data/dreamsim_pipeline.py
python scripts/data/dreamsim_pipeline.py --keys murty185 nsd
```

## Generate DreamSim Cluster Outputs

KMeans/euclidean:

```bash
python scripts/data/cluster_runner.py dreamsim \
  --pth data/dreamsim/dreamsim_embeddings.pth \
  --embedding_key murty185 \
  --image_dir public/images \
  --out_dir public/data/dreamsim_clusters
```

Hierarchical/correlation:

```bash
python scripts/data/cluster_runner.py dreamsim-hierarchical \
  --pth data/dreamsim/dreamsim_embeddings.pth \
  --embedding_key murty185 \
  --image_dir public/images \
  --out_dir public/data/dreamsim_clusters_hierarchical_corr
```
