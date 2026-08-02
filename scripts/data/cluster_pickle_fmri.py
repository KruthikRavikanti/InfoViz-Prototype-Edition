import argparse
import inspect
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_numeric_array(x: Any) -> bool:
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    return np.issubdtype(arr.dtype, np.number)


def tsne_iteration_kwargs() -> Dict[str, int]:
    tsne_sig = inspect.signature(TSNE.__init__)
    if "max_iter" in tsne_sig.parameters:
        return {"max_iter": 1000}
    return {"n_iter": 1000}


def safe_tsne(matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    n = matrix.shape[0]
    perplexity = max(2, min(30, (n - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        **tsne_iteration_kwargs(),
    )
    return tsne.fit_transform(matrix)


def safe_tsne_precomputed(distance_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    n = distance_matrix.shape[0]
    perplexity = max(2, min(30, (n - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        metric="precomputed",
        random_state=random_state,
        **tsne_iteration_kwargs(),
    )
    return tsne.fit_transform(distance_matrix)


def choose_best_k(points: np.ndarray, k_min: int = 2, k_max: int = 5, random_state: int = 42):
    n = len(points)
    best = None

    for k in range(k_min, k_max + 1):
        if n < max(k * 3, 6):
            continue

        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(points)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(points, labels)

        if best is None or score > best["silhouette"]:
            best = {
                "k": k,
                "labels": labels,
                "silhouette": float(score),
                "model": km,
            }

    return best


def choose_best_hierarchical_k(
    distance_vector: np.ndarray,
    linkage_matrix: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
):
    distance_matrix = squareform(distance_vector)
    best = None

    for k in range(k_min, k_max + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        if best is None or score > best["silhouette"]:
            best = {"k": k, "labels": labels, "silhouette": float(score)}

    return best


def cluster_size_dict(labels: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def clean_correlation_distance_vector(distance_vector: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(distance_vector, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(cleaned, 0.0, 2.0)


def distance_to_cluster_medoids(distance_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.zeros(len(labels), dtype=float)

    for label in np.unique(labels):
        member_indices = np.flatnonzero(labels == label)
        if len(member_indices) == 1:
            distances[member_indices[0]] = 0.0
            continue

        within_cluster = distance_matrix[np.ix_(member_indices, member_indices)]
        medoid_local_index = int(np.argmin(within_cluster.sum(axis=1)))
        medoid_index = member_indices[medoid_local_index]
        distances[member_indices] = distance_matrix[member_indices, medoid_index]

    return distances


def save_dendrogram_png(
    linkage_matrix: np.ndarray,
    image_ids: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 8))
    dendrogram(
        linkage_matrix,
        labels=image_ids.astype(str).tolist(),
        leaf_rotation=90,
        leaf_font_size=6,
    )
    plt.title(title)
    plt.xlabel("Image")
    plt.ylabel("Correlation distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def flatten_prefix(prefix: Tuple[str, ...]) -> str:
    return "__".join(prefix) if prefix else "root"


def try_make_image_ids(
    n_rows: int,
    data_dict: Optional[dict],
    preferred_keys: Iterable[str] = ("image_names", "filenames", "image_ids", "stimulus_ids"),
):
    if isinstance(data_dict, dict):
        for key in preferred_keys:
            if key in data_dict:
                vals = data_dict[key]
                if isinstance(vals, (list, tuple, np.ndarray)) and len(vals) == n_rows:
                    return pd.Series([str(x) for x in vals], name="image_name")
    return pd.Series([f"image_{i+1:04d}" for i in range(n_rows)], name="image_name")


def process_array_candidate(arr: np.ndarray, parent_dict: Optional[dict], prefix: Tuple[str, ...]):
    arr = np.asarray(arr)

    if arr.ndim == 1:
        return None

    if arr.ndim == 2:
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        image_name = try_make_image_ids(arr.shape[0], parent_dict)
        return {
            "name": flatten_prefix(prefix),
            "matrix": arr,
            "image_name": image_name,
            "meta": {"original_shape": list(arr.shape), "reduction": "none"},
        }

    if arr.ndim == 3:
        shape = arr.shape
        smallest_axis = int(np.argmin(shape))
        reduced = arr.mean(axis=smallest_axis)

        if reduced.ndim != 2:
            return None

        if reduced.shape[0] > reduced.shape[1]:
            reduced = reduced.T

        if reduced.shape[0] < 2 or reduced.shape[1] < 2:
            return None

        image_name = try_make_image_ids(reduced.shape[0], parent_dict)
        return {
            "name": flatten_prefix(prefix),
            "matrix": reduced,
            "image_name": image_name,
            "meta": {
                "original_shape": list(shape),
                "reduction": f"mean_over_axis_{smallest_axis}",
            },
        }

    return None


def extract_candidate_matrices(obj: Any, prefix: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 0 and len(df) > 1:
            image_name = None
            for col in df.columns:
                if col not in numeric_df.columns:
                    image_name = pd.Series(df[col].astype(str).values, name="image_name")
                    break
            if image_name is None:
                image_name = pd.Series([f"image_{i+1:04d}" for i in range(len(df))], name="image_name")
            candidates.append({
                "name": flatten_prefix(prefix),
                "matrix": numeric_df.to_numpy(),
                "image_name": image_name,
                "meta": {"source_type": "dataframe"},
            })
        return candidates

    if isinstance(obj, dict):
        matrix_keys = ["data", "responses", "fmri", "X", "matrix", "voxels", "activations"]
        for key in matrix_keys:
            if key in obj and is_numeric_array(obj[key]):
                processed = process_array_candidate(np.asarray(obj[key]), obj, prefix + (key,))
                if processed is not None:
                    candidates.append(processed)

        for key, value in obj.items():
            if key in matrix_keys:
                continue
            candidates.extend(extract_candidate_matrices(value, prefix + (str(key),)))
        return candidates

    if is_numeric_array(obj):
        processed = process_array_candidate(np.asarray(obj), None, prefix)
        if processed is not None:
            candidates.append(processed)

    return candidates


def cluster_matrix(
    matrix: np.ndarray,
    image_name: pd.Series,
    out_dir: Path,
    base_name: str,
    random_state: int = 42,
    pca_dim: int = 50,
):
    X = np.asarray(matrix, dtype=float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    Xz = StandardScaler().fit_transform(X)

    used_pca_dim = max(2, min(pca_dim, Xz.shape[0] - 1, Xz.shape[1]))
    reducer = PCA(n_components=used_pca_dim, random_state=random_state)
    X_reduced = reducer.fit_transform(Xz)

    best = choose_best_k(X_reduced, k_min=2, k_max=5, random_state=random_state)
    if best is None:
        raise ValueError(f"Could not find a valid clustering for {base_name}")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne(X_reduced, random_state=random_state)

    result_df = pd.DataFrame({
        "image_name": image_name.values,
        "cluster_label": best["labels"],
        "pca_x": pca2[:, 0],
        "pca_y": pca2[:, 1],
        "tsne_x": tsne2[:, 0],
        "tsne_y": tsne2[:, 1],
        "distance_to_centroid": np.linalg.norm(
            X_reduced - best["model"].cluster_centers_[best["labels"]],
            axis=1,
        ),
    })

    result_csv = out_dir / f"{base_name}_clusters.csv"
    result_df.to_csv(result_csv, index=False)

    summary = {
        "name": base_name,
        "result_csv": str(result_csv),
        "n_images": int(X.shape[0]),
        "n_voxels": int(X.shape[1]),
        "pca_dim_used": int(used_pca_dim),
        "clustering_method": "kmeans",
        "distance_metric": "euclidean",
        "normalization": "voxel_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "explained_variance_ratio_first10": reducer.explained_variance_ratio_[:10].tolist(),
    }

    with open(out_dir / f"{base_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def cluster_matrix_hierarchical_corr(
    matrix: np.ndarray,
    image_name: pd.Series,
    out_dir: Path,
    base_name: str,
    random_state: int = 42,
    linkage_method: str = "average",
    k_min: int = 2,
    k_max: int = 5,
    save_dendrogram: bool = True,
):
    X = np.asarray(matrix, dtype=float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    Xz = StandardScaler().fit_transform(X)

    distance_vector = clean_correlation_distance_vector(pdist(Xz, metric="correlation"))
    distance_matrix = squareform(distance_vector)
    linkage_matrix = linkage(distance_vector, method=linkage_method)

    best = choose_best_hierarchical_k(distance_vector, linkage_matrix, k_min=k_min, k_max=k_max)
    if best is None:
        raise ValueError(f"Could not find a valid hierarchical clustering for {base_name}")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne_precomputed(distance_matrix, random_state=random_state)
    medoid_distances = distance_to_cluster_medoids(distance_matrix, best["labels"])

    result_csv = out_dir / f"{base_name}_clusters.csv"
    linkage_csv = out_dir / f"{base_name}_linkage.csv"
    dendrogram_path = out_dir / f"{base_name}_dendrogram.png"

    result_df = pd.DataFrame({
        "image_name": image_name.values,
        "cluster_label": best["labels"],
        "pca_x": pca2[:, 0],
        "pca_y": pca2[:, 1],
        "plot_x": pca2[:, 0],
        "plot_y": pca2[:, 1],
        "tsne_x": tsne2[:, 0],
        "tsne_y": tsne2[:, 1],
        "distance_to_centroid": medoid_distances,
        "distance_to_medoid": medoid_distances,
    })
    result_df.to_csv(result_csv, index=False)

    pd.DataFrame(
        linkage_matrix,
        columns=["child_1", "child_2", "distance", "n_observations"],
    ).to_csv(linkage_csv, index=False)

    if save_dendrogram:
        save_dendrogram_png(
            linkage_matrix,
            image_name,
            dendrogram_path,
            title=f"{base_name} hierarchical correlation dendrogram",
        )

    summary = {
        "name": base_name,
        "result_csv": str(result_csv),
        "n_images": int(X.shape[0]),
        "n_voxels": int(X.shape[1]),
        "clustering_method": "hierarchical",
        "distance_metric": "correlation",
        "linkage_method": linkage_method,
        "normalization": "voxel_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "linkage_csv": str(linkage_csv),
        "dendrogram_png": str(dendrogram_path) if save_dendrogram else None,
        "distance_to_cluster_representative": "medoid_correlation_distance",
    }

    with open(out_dir / f"{base_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Cluster real fMRI ground-truth data from a pickle file using the same pipeline as the prediction clustering."
    )
    parser.add_argument("--pickle_path", type=Path, required=True, help="Path to .pickle or .pkl file")
    parser.add_argument("--out_dir", type=Path, required=True, help="Folder for outputs")
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--method",
        choices=["kmeans", "hierarchical-correlation"],
        default="kmeans",
        help="Clustering method to apply to each extracted matrix.",
    )
    parser.add_argument(
        "--linkage_method",
        choices=["single", "complete", "average", "weighted"],
        default="average",
    )
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--skip_dendrogram", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    with open(args.pickle_path, "rb") as f:
        obj = pickle.load(f)

    candidates = extract_candidate_matrices(obj)
    if not candidates:
        raise ValueError(
            "No usable numeric matrices were found in the pickle. "
            "You may need to inspect the pickle schema and adjust extraction logic."
        )

    summaries = []
    manifest = []

    for idx, cand in enumerate(candidates):
        base_name = cand["name"].replace("/", "__").replace(" ", "_")
        print(f"Clustering candidate {idx+1}/{len(candidates)}: {base_name} | shape={cand['matrix'].shape}")

        if args.method == "hierarchical-correlation":
            summary = cluster_matrix_hierarchical_corr(
                matrix=cand["matrix"],
                image_name=cand["image_name"],
                out_dir=args.out_dir,
                base_name=base_name,
                random_state=args.random_state,
                linkage_method=args.linkage_method,
                k_min=args.k_min,
                k_max=args.k_max,
                save_dendrogram=not args.skip_dendrogram,
            )
        else:
            summary = cluster_matrix(
                matrix=cand["matrix"],
                image_name=cand["image_name"],
                out_dir=args.out_dir,
                base_name=base_name,
                random_state=args.random_state,
                pca_dim=args.pca_dim,
            )
        summary["meta"] = cand.get("meta", {})
        summaries.append(summary)

        manifest.append({
            "name": base_name,
            "shape": list(np.asarray(cand["matrix"]).shape),
            "meta": cand.get("meta", {}),
        })

    pd.DataFrame(summaries).to_csv(args.out_dir / "pickle_cluster_summary.csv", index=False)

    with open(args.out_dir / "pickle_cluster_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    with open(args.out_dir / "pickle_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
