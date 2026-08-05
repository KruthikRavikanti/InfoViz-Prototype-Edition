"""
Cluster prediction CSVs and image-derived visual features.

Mentor guide:
- `voxel`: original voxel clustering, using z-scored voxels + PCA + KMeans/euclidean.
- `voxel-hierarchical`: z-scored voxels + correlation distance + hierarchical clustering.
- `visual`: original visual-feature clustering, using z-scored image features + KMeans/euclidean.
- `visual-hierarchical`: z-scored image features + correlation distance + hierarchical clustering.
- `dreamsim`: DreamSim embeddings + PCA + KMeans/euclidean.
- `dreamsim-hierarchical`: DreamSim embeddings + correlation distance + hierarchical clustering.

Ground-truth fMRI pickle clustering uses the same method ideas in `cluster_pickle_fmri.py`.
"""

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ID_CANDIDATES = [
    "image_name",
    "filename",
    "file_name",
    "image",
    "image_id",
    "img",
    "img_id",
    "stimulus",
    "stimulus_id",
]

FILENAME_RE = re.compile(
    r"^murtylab_(?P<model>.+?)_nsd_1000_(?P<roi>ffa|ppa|eba)_(?P<timestamp>\d+Z)\.csv$",
    re.IGNORECASE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def artifact_path(path: Path) -> str:
    """Store repo-local artifact paths without machine-specific absolute prefixes."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def detect_id_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in ID_CANDIDATES:
        if c in lower_map:
            return lower_map[c]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in df.columns:
        if col not in numeric_cols and not str(col).startswith("Unnamed"):
            return col
    return df.columns[0]


def parse_model_roi_from_filename(path: Path) -> Tuple[Optional[str], Optional[str]]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None, None
    return m.group("model"), m.group("roi")


# ---------------------------------------------------------------------------
# Shared clustering helpers
# ---------------------------------------------------------------------------

def choose_best_k(points: np.ndarray, k_min: int = 2, k_max: int = 5, random_state: int = 42):
    """KMeans/euclidean path: try k values and keep the best silhouette score."""
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
            best = {"k": k, "labels": labels, "silhouette": float(score), "model": km}
    return best


def choose_best_hierarchical_k(
    distance_vector: np.ndarray,
    linkage_matrix: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
):
    """Hierarchical/correlation path: cut the dendrogram at k values and score each cut."""
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


def tsne_iteration_kwargs() -> Dict[str, int]:
    """Support both older sklearn `n_iter` and newer sklearn `max_iter` TSNE APIs."""
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


def cluster_size_dict(labels: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def prepare_voxel_matrix(df: pd.DataFrame, id_col: str) -> Tuple[pd.Series, pd.DataFrame]:
    """Return image IDs and the numeric voxel matrix. Rows are images; columns are voxels."""
    work = df.copy()
    drop_cols = [c for c in work.columns if str(c).startswith("Unnamed")]
    if id_col in work.columns:
        drop_cols.append(id_col)
    image_ids = work[id_col].astype(str).copy()
    voxel_df = work.drop(columns=drop_cols, errors="ignore")
    voxel_df = voxel_df.apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in voxel_df.columns if voxel_df[c].notna().sum() > 0]
    voxel_df = voxel_df[valid_cols]
    if voxel_df.shape[1] == 0:
        raise ValueError("No numeric voxel columns were found after cleaning.")
    return image_ids, voxel_df


def clean_correlation_distance_vector(distance_vector: np.ndarray) -> np.ndarray:
    """Correlation distance can be nan for constant rows; replace with a neutral distance."""
    cleaned = np.nan_to_num(distance_vector, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(cleaned, 0.0, 2.0)


def distance_to_cluster_medoids(distance_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Hierarchical clusters have no centroid, so use the cluster medoid as representative."""
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
    """Save the full hierarchical tree for mentor inspection."""
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


# ---------------------------------------------------------------------------
# Voxel clustering modes
# ---------------------------------------------------------------------------

def cluster_single_voxel_csv(csv_path: Path, out_dir: Path, pca_dim: int = 50, random_state: int = 42) -> Dict:
    """Original voxel method: impute, z-score voxels, PCA, then KMeans/euclidean."""
    df = pd.read_csv(csv_path)
    id_col = detect_id_column(df)
    image_ids, voxel_df = prepare_voxel_matrix(df, id_col)

    # Z-score happens per voxel column across images.
    X = SimpleImputer(strategy="mean").fit_transform(voxel_df.values)
    Xz = StandardScaler().fit_transform(X)

    # KMeans is run in PCA space to reduce high-dimensional voxel noise.
    used_pca_dim = max(2, min(pca_dim, Xz.shape[0] - 1, Xz.shape[1]))
    reducer = PCA(n_components=used_pca_dim, random_state=random_state)
    X_reduced = reducer.fit_transform(Xz)

    best = choose_best_k(X_reduced, k_min=2, k_max=5, random_state=random_state)
    if best is None:
        raise ValueError(f"Could not find a valid clustering for {csv_path.name}")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne(X_reduced, random_state=random_state)

    model_name, roi = parse_model_roi_from_filename(csv_path)

    result_df = pd.DataFrame({
        "image_name": image_ids.values,
        "source_file": csv_path.name,
        "model": model_name if model_name is not None else "",
        "roi": roi if roi is not None else "",
        "cluster_label": best["labels"],
        "pca_x": pca2[:, 0],
        "pca_y": pca2[:, 1],
        "tsne_x": tsne2[:, 0],
        "tsne_y": tsne2[:, 1],
        "distance_to_centroid": np.linalg.norm(X_reduced - best["model"].cluster_centers_[best["labels"]], axis=1),
    })

    base = csv_path.stem
    result_csv = out_dir / f"{base}_clusters.csv"
    result_df.to_csv(result_csv, index=False)

    summary = {
        "source_file": csv_path.name,
        "result_csv": artifact_path(result_csv),
        "model": model_name,
        "roi": roi,
        "n_images": int(X.shape[0]),
        "n_voxels": int(X.shape[1]),
        "pca_dim_used": int(used_pca_dim),
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "explained_variance_ratio_first10": reducer.explained_variance_ratio_[:10].tolist(),
    }

    with open(out_dir / f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def cluster_single_voxel_csv_hierarchical_corr(
    csv_path: Path,
    out_dir: Path,
    linkage_method: str = "average",
    k_min: int = 2,
    k_max: int = 5,
    random_state: int = 42,
    save_dendrogram: bool = True,
) -> Dict:
    """New voxel method: z-score voxels, then cluster pairwise correlation distances."""
    df = pd.read_csv(csv_path)
    id_col = detect_id_column(df)
    image_ids, voxel_df = prepare_voxel_matrix(df, id_col)

    X = SimpleImputer(strategy="mean").fit_transform(voxel_df.values)
    Xz = StandardScaler().fit_transform(X)

    # `pdist(..., metric="correlation")` calculates 1 - corr(row_i, row_j).
    distance_vector = clean_correlation_distance_vector(pdist(Xz, metric="correlation"))
    distance_matrix = squareform(distance_vector)
    linkage_matrix = linkage(distance_vector, method=linkage_method)

    best = choose_best_hierarchical_k(distance_vector, linkage_matrix, k_min=k_min, k_max=k_max)
    if best is None:
        raise ValueError(f"Could not find a valid hierarchical clustering for {csv_path.name}")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne_precomputed(distance_matrix, random_state=random_state)
    # Stored as distance_to_centroid too so existing CSV readers still work.
    medoid_distances = distance_to_cluster_medoids(distance_matrix, best["labels"])

    model_name, roi = parse_model_roi_from_filename(csv_path)
    base = csv_path.stem
    result_csv = out_dir / f"{base}_clusters.csv"
    linkage_csv = out_dir / f"{base}_linkage.csv"
    dendrogram_path = out_dir / f"{base}_dendrogram.png"

    result_df = pd.DataFrame({
        "image_name": image_ids.values,
        "source_file": csv_path.name,
        "model": model_name if model_name is not None else "",
        "roi": roi if roi is not None else "",
        "cluster_label": best["labels"],
        "pca_x": pca2[:, 0],
        "pca_y": pca2[:, 1],
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
            image_ids,
            dendrogram_path,
            title=f"{model_name or base} {roi or ''} hierarchical correlation dendrogram".strip(),
        )

    summary = {
        "source_file": csv_path.name,
        "result_csv": artifact_path(result_csv),
        "model": model_name,
        "roi": roi,
        "n_images": int(X.shape[0]),
        "n_voxels": int(X.shape[1]),
        "clustering_method": "hierarchical",
        "distance_metric": "correlation",
        "linkage_method": linkage_method,
        "normalization": "voxel_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "linkage_csv": artifact_path(linkage_csv),
        "dendrogram_png": artifact_path(dendrogram_path) if save_dendrogram else None,
        "distance_to_cluster_representative": "medoid_correlation_distance",
    }

    with open(out_dir / f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_voxel_mode(input_dir: Path, out_dir: Path, pca_dim: int, random_state: int) -> None:
    ensure_dir(out_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    summaries = []
    for path in csv_files:
        print(f"Clustering voxel file: {path.name}")
        summaries.append(cluster_single_voxel_csv(path, out_dir, pca_dim=pca_dim, random_state=random_state))

    pd.DataFrame(summaries).to_csv(out_dir / "voxel_cluster_summary.csv", index=False)
    with open(out_dir / "voxel_cluster_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved voxel outputs to: {out_dir}")


def run_voxel_hierarchical_mode(
    input_dir: Path,
    out_dir: Path,
    linkage_method: str,
    k_min: int,
    k_max: int,
    random_state: int,
    save_dendrogram: bool,
) -> None:
    ensure_dir(out_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    summaries = []
    for path in csv_files:
        print(f"Hierarchical correlation clustering voxel file: {path.name}")
        summaries.append(
            cluster_single_voxel_csv_hierarchical_corr(
                path,
                out_dir,
                linkage_method=linkage_method,
                k_min=k_min,
                k_max=k_max,
                random_state=random_state,
                save_dendrogram=save_dendrogram,
            )
        )

    pd.DataFrame(summaries).to_csv(out_dir / "voxel_cluster_summary.csv", index=False)
    with open(out_dir / "voxel_cluster_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved hierarchical correlation voxel outputs to: {out_dir}")


# ---------------------------------------------------------------------------
# Visual feature extraction
# ---------------------------------------------------------------------------

def rgb_to_hsl_np(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    l = (maxc + minc) / 2.0

    d = maxc - minc
    s = np.zeros_like(l)
    nonzero = d > 1e-12
    s1 = d / np.clip(maxc + minc, 1e-12, None)
    s2 = d / np.clip(2.0 - maxc - minc, 1e-12, None)
    s = np.where(nonzero & (l <= 0.5), s1, s)
    s = np.where(nonzero & (l > 0.5), s2, s)

    h = np.zeros_like(l)
    rc = (((g - b) / np.clip(d, 1e-12, None)) % 6.0)
    gc = ((b - r) / np.clip(d, 1e-12, None)) + 2.0
    bc = ((r - g) / np.clip(d, 1e-12, None)) + 4.0

    is_r = nonzero & (maxc == r)
    is_g = nonzero & (maxc == g)
    is_b = nonzero & (maxc == b)

    h = np.where(is_r, rc, h)
    h = np.where(is_g, gc, h)
    h = np.where(is_b, bc, h)
    h = (h / 6.0) * 360.0

    return h, s, l


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.deg2rad(values)
    s = np.mean(np.sin(radians))
    c = np.mean(np.cos(radians))
    angle = np.rad2deg(np.arctan2(s, c))
    return float(angle + 360.0 if angle < 0 else angle)


def circular_std_deg(values: np.ndarray) -> float:
    radians = np.deg2rad(values)
    R = np.sqrt(np.mean(np.sin(radians)) ** 2 + np.mean(np.cos(radians)) ** 2)
    R = max(float(R), 1e-10)
    return float(np.sqrt(-2.0 * np.log(R)) * (180.0 / np.pi))


def shannon_entropy(values: np.ndarray, bins: int, min_val: float, max_val: float) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(min_val, max_val))
    probs = hist.astype(np.float64)
    probs = probs[probs > 0]
    probs = probs / probs.sum()
    return float(-(probs * np.log2(probs)).sum())


def sobel_features(gray: np.ndarray) -> Tuple[float, float, float, float, float]:
    gx = (
        -gray[:-2, :-2] - 2 * gray[1:-1, :-2] - gray[2:, :-2]
        + gray[:-2, 2:] + 2 * gray[1:-1, 2:] + gray[2:, 2:]
    )
    gy = (
        -gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - gray[:-2, 2:]
        + gray[2:, :-2] + 2 * gray[2:, 1:-1] + gray[2:, 2:]
    )
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mask = mag > 0.15
    edge_density = float(edge_mask.mean())
    if edge_mask.sum() == 0:
        return edge_density, 0.0, 0.0, 0.0, 0.0

    angle_abs = np.rad2deg(np.arctan2(np.abs(gy[edge_mask]), np.abs(gx[edge_mask])))
    pct_horizontal = float((angle_abs < 20).mean())
    pct_vertical = float((angle_abs > 70).mean())
    pct_diagonal = float(((angle_abs >= 20) & (angle_abs <= 70)).mean())

    full_angle = np.rad2deg(np.arctan2(gy[edge_mask], gx[edge_mask]))
    full_angle = np.where(full_angle < 0, full_angle + 180, full_angle)
    hist, _ = np.histogram(full_angle, bins=180, range=(0, 180))
    dominant_angle = float(np.argmax(hist))
    return edge_density, pct_horizontal, pct_vertical, pct_diagonal, dominant_angle


def prepare_image(path: Path, max_dim: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = min(max_dim / w, max_dim / h)
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    return np.asarray(img)


def extract_basic_features(img_arr: np.ndarray) -> Dict[str, float]:
    """Small feature set: image-level brightness, saturation, and hue summaries."""
    hue, sat, light = rgb_to_hsl_np(img_arr)
    light_flat = light.reshape(-1)
    sat_flat = sat.reshape(-1)
    hue_flat = hue.reshape(-1)
    return {
        "brightness_median": float(np.median(light_flat)),
        "brightness_stdev": float(np.std(light_flat)),
        "saturation_median": float(np.median(sat_flat)),
        "saturation_stdev": float(np.std(sat_flat)),
        "hue_median": circular_mean_deg(hue_flat),
        "hue_stdev": circular_std_deg(hue_flat),
    }


def extract_all_features(img_arr: np.ndarray) -> Dict[str, float]:
    """Full visual feature set used by the visual clustering modes."""
    features = extract_basic_features(img_arr)
    h, s, l = rgb_to_hsl_np(img_arr)
    height, width = l.shape

    grid_labels = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]
    cell_h = height // 3
    cell_w = width // 3
    idx = 0
    for gr in range(3):
        for gc in range(3):
            y0 = gr * cell_h
            y1 = height if gr == 2 else (gr + 1) * cell_h
            x0 = gc * cell_w
            x1 = width if gc == 2 else (gc + 1) * cell_w
            l_cell = l[y0:y1, x0:x1].reshape(-1)
            s_cell = s[y0:y1, x0:x1].reshape(-1)
            h_cell = h[y0:y1, x0:x1].reshape(-1)
            label = grid_labels[idx]
            features[f"grid_{label}_brightness"] = float(l_cell.mean())
            features[f"grid_{label}_saturation"] = float(s_cell.mean())
            features[f"grid_{label}_hue"] = circular_mean_deg(h_cell)
            idx += 1

    light_flat = l.reshape(-1)
    sat_flat = s.reshape(-1)
    hue_flat = h.reshape(-1)
    features["brightness_entropy"] = shannon_entropy(light_flat, bins=32, min_val=0.0, max_val=1.0)
    features["hue_entropy"] = shannon_entropy(hue_flat, bins=36, min_val=0.0, max_val=360.0)

    edge_density, pct_h, pct_v, pct_d, dom_angle = sobel_features(l.astype(np.float32))
    features["edge_density"] = edge_density
    features["pct_horizontal"] = pct_h
    features["pct_vertical"] = pct_v
    features["pct_diagonal"] = pct_d
    features["dominant_angle"] = dom_angle

    gray_hist, _ = np.histogram(light_flat, bins=16, range=(0.0, 1.0))
    gray_hist = gray_hist / gray_hist.sum()
    for i, value in enumerate(gray_hist):
        features[f"gray_hist_{i:02d}"] = float(value)

    sat_hist, _ = np.histogram(sat_flat, bins=16, range=(0.0, 1.0))
    sat_hist = sat_hist / sat_hist.sum()
    for i, value in enumerate(sat_hist):
        features[f"sat_hist_{i:02d}"] = float(value)

    gray = l.astype(np.float32)
    if height > 1 and width > 1:
        local_contrast = (
            np.abs(gray[:, 1:] - gray[:, :-1]).mean() +
            np.abs(gray[1:, :] - gray[:-1, :]).mean()
        ) / 2.0
    else:
        local_contrast = 0.0
    features["local_contrast"] = float(local_contrast)
    features["aspect_ratio"] = float(width / max(height, 1))

    rgb = img_arr.astype(np.float32) / 255.0
    features["red_mean"] = float(rgb[..., 0].mean())
    features["green_mean"] = float(rgb[..., 1].mean())
    features["blue_mean"] = float(rgb[..., 2].mean())
    return features


def list_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def extract_visual_feature_frame(image_dir: Path, feature_set: str) -> pd.DataFrame:
    """Convert the image folder into an image-by-feature table."""
    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    rows = []
    for path in image_paths:
        print(f"Extracting features: {path.name}")
        img = prepare_image(path, max_dim=256)
        feats = extract_basic_features(img) if feature_set == "basic" else extract_all_features(img)
        row = {"image_name": path.name}
        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DreamSim embedding helpers
# ---------------------------------------------------------------------------

def load_dreamsim_embedding_matrix(pth_path: Path, key: str) -> np.ndarray:
    """Load a named tensor from the DreamSim .pth bundle as a numpy matrix."""
    import torch

    if not pth_path.is_file():
        raise FileNotFoundError(f"DreamSim embedding file not found: {pth_path}")

    try:
        blob = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(str(pth_path), map_location="cpu")

    if not isinstance(blob, dict):
        raise TypeError(f"Expected dict[str, Tensor] in {pth_path}, got {type(blob)}")
    if key not in blob:
        raise KeyError(f"Embedding key {key!r} not found. Available keys: {sorted(blob.keys())}")

    tensor = blob[key]
    if not hasattr(tensor, "detach"):
        raise TypeError(f"Embedding key {key!r} is not a torch Tensor.")

    matrix = tensor.detach().cpu().float().numpy()
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix for {key!r}, got shape {matrix.shape}")

    return matrix


def dreamsim_image_names(image_dir: Optional[Path], n_images: int) -> pd.Series:
    """
    Map DreamSim rows to app image filenames.

    For this repo, murty185 embeddings are in stimulus order and line up with
    public/images/image_001.png ... image_185.png.
    """
    if image_dir is not None:
        image_paths = list_images(image_dir)
        if len(image_paths) != n_images:
            raise ValueError(
                f"DreamSim rows ({n_images}) do not match image files in {image_dir} ({len(image_paths)})."
            )
        return pd.Series([path.name for path in image_paths])

    return pd.Series([f"image_{i:03d}.png" for i in range(1, n_images + 1)])


def cluster_dreamsim_embeddings(
    pth_path: Path,
    out_dir: Path,
    embedding_key: str,
    image_dir: Optional[Path],
    pca_dim: int,
    random_state: int,
) -> Dict:
    """DreamSim KMeans/euclidean method: z-score embeddings, PCA, then KMeans."""
    ensure_dir(out_dir)

    X = load_dreamsim_embedding_matrix(pth_path, embedding_key)
    image_ids = dreamsim_image_names(image_dir, X.shape[0])
    Xz = StandardScaler().fit_transform(X)

    used_pca_dim = max(2, min(pca_dim, Xz.shape[0] - 1, Xz.shape[1]))
    reducer = PCA(n_components=used_pca_dim, random_state=random_state)
    X_reduced = reducer.fit_transform(Xz)

    best = choose_best_k(X_reduced, k_min=2, k_max=5, random_state=random_state)
    if best is None:
        raise ValueError(f"Could not find a valid DreamSim clustering for key {embedding_key!r}.")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne(X_reduced, random_state=random_state)

    result_csv = out_dir / "dreamsim_clusters.csv"
    result_df = pd.DataFrame({
        "image_name": image_ids.values,
        "cluster_label": best["labels"],
        "embedding_key": embedding_key,
        "clustering_method": "kmeans",
        "distance_metric": "euclidean",
        "plot_x": pca2[:, 0],
        "plot_y": pca2[:, 1],
        "pca_x": pca2[:, 0],
        "pca_y": pca2[:, 1],
        "tsne_x": tsne2[:, 0],
        "tsne_y": tsne2[:, 1],
        "distance_to_centroid": np.linalg.norm(X_reduced - best["model"].cluster_centers_[best["labels"]], axis=1),
    })
    result_df.to_csv(result_csv, index=False)

    summary = {
        "source_file": artifact_path(pth_path),
        "result_csv": artifact_path(result_csv),
        "n_images": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "embedding_key": embedding_key,
        "feature_set": f"dreamsim_{embedding_key}_embeddings",
        "pca_dim_used": int(used_pca_dim),
        "projection_used_for_clustering": f"pca_{used_pca_dim}",
        "clustering_method": "kmeans",
        "distance_metric": "euclidean",
        "normalization": "embedding_feature_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "explained_variance_ratio_first10": reducer.explained_variance_ratio_[:10].tolist(),
    }

    with open(out_dir / "dreamsim_cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved DreamSim outputs to: {out_dir}")
    return summary


def cluster_dreamsim_embeddings_hierarchical_corr(
    pth_path: Path,
    out_dir: Path,
    embedding_key: str,
    image_dir: Optional[Path],
    linkage_method: str,
    k_min: int,
    k_max: int,
    random_state: int,
    save_dendrogram: bool,
) -> Dict:
    """DreamSim hierarchical method: z-score embeddings, then cluster correlation distances."""
    ensure_dir(out_dir)

    X = load_dreamsim_embedding_matrix(pth_path, embedding_key)
    image_ids = dreamsim_image_names(image_dir, X.shape[0])
    Xz = StandardScaler().fit_transform(X)

    distance_vector = clean_correlation_distance_vector(pdist(Xz, metric="correlation"))
    distance_matrix = squareform(distance_vector)
    linkage_matrix = linkage(distance_vector, method=linkage_method)

    best = choose_best_hierarchical_k(distance_vector, linkage_matrix, k_min=k_min, k_max=k_max)
    if best is None:
        raise ValueError(f"Could not find a valid DreamSim hierarchical clustering for key {embedding_key!r}.")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne_precomputed(distance_matrix, random_state=random_state)
    medoid_distances = distance_to_cluster_medoids(distance_matrix, best["labels"])

    result_csv = out_dir / "dreamsim_clusters.csv"
    linkage_csv = out_dir / "dreamsim_linkage.csv"
    dendrogram_path = out_dir / "dreamsim_dendrogram.png"

    result_df = pd.DataFrame({
        "image_name": image_ids.values,
        "cluster_label": best["labels"],
        "embedding_key": embedding_key,
        "clustering_method": "hierarchical",
        "distance_metric": "correlation",
        "linkage_method": linkage_method,
        "plot_x": pca2[:, 0],
        "plot_y": pca2[:, 1],
        "pca_x": pca2[:, 0],
        "pca_y": pca2[:, 1],
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
            image_ids,
            dendrogram_path,
            title=f"DreamSim {embedding_key} hierarchical correlation dendrogram",
        )

    summary = {
        "source_file": artifact_path(pth_path),
        "result_csv": artifact_path(result_csv),
        "n_images": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "embedding_key": embedding_key,
        "feature_set": f"dreamsim_{embedding_key}_embeddings",
        "projection_used_for_clustering": "correlation_distance",
        "clustering_method": "hierarchical",
        "distance_metric": "correlation",
        "linkage_method": linkage_method,
        "normalization": "embedding_feature_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "linkage_csv": artifact_path(linkage_csv),
        "dendrogram_png": artifact_path(dendrogram_path) if save_dendrogram else None,
        "distance_to_cluster_representative": "medoid_correlation_distance",
    }

    with open(out_dir / "dreamsim_cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved hierarchical correlation DreamSim outputs to: {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# Visual clustering modes
# ---------------------------------------------------------------------------

def run_visual_mode(image_dir: Path, out_dir: Path, feature_set: str, projection: str, random_state: int) -> None:
    """Original visual method: z-score visual features, project, then KMeans/euclidean."""
    ensure_dir(out_dir)

    feat_df = extract_visual_feature_frame(image_dir, feature_set)
    feature_cols = [c for c in feat_df.columns if c != "image_name"]
    X = feat_df[feature_cols].values
    Xz = StandardScaler().fit_transform(X)

    # For visual KMeans, the selected 2D projection is the clustering space.
    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    if projection == "pca":
        cluster_space = pca2
        plot_x, plot_y = pca2[:, 0], pca2[:, 1]
        tsne_x = np.full(len(feat_df), np.nan)
        tsne_y = np.full(len(feat_df), np.nan)
    else:
        tsne2 = safe_tsne(Xz, random_state=random_state)
        cluster_space = tsne2
        plot_x, plot_y = tsne2[:, 0], tsne2[:, 1]
        tsne_x, tsne_y = tsne2[:, 0], tsne2[:, 1]

    best = choose_best_k(cluster_space, k_min=2, k_max=5, random_state=random_state)
    if best is None:
        raise ValueError("Could not find a valid visual clustering.")

    result_df = feat_df.copy()
    result_df["cluster_label"] = best["labels"]
    result_df["feature_set"] = feature_set
    result_df["projection_used_for_clustering"] = projection
    result_df["plot_x"] = plot_x
    result_df["plot_y"] = plot_y
    result_df["pca_x"] = pca2[:, 0]
    result_df["pca_y"] = pca2[:, 1]
    result_df["tsne_x"] = tsne_x
    result_df["tsne_y"] = tsne_y
    result_df.to_csv(out_dir / "visual_clusters.csv", index=False)
    feat_df.to_csv(out_dir / "visual_features_only.csv", index=False)

    summary = {
        "n_images": int(len(feat_df)),
        "result_csv": artifact_path(out_dir / "visual_clusters.csv"),
        "feature_set": feature_set,
        "projection_used_for_clustering": projection,
        "clustering_method": "kmeans",
        "distance_metric": "euclidean",
        "normalization": "feature_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "feature_columns": feature_cols,
    }
    with open(out_dir / "visual_cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved visual outputs to: {out_dir}")


def run_visual_hierarchical_mode(
    image_dir: Path,
    out_dir: Path,
    feature_set: str,
    linkage_method: str,
    k_min: int,
    k_max: int,
    random_state: int,
    save_dendrogram: bool,
) -> None:
    """New visual method: z-score visual features, then cluster correlation distances."""
    ensure_dir(out_dir)

    feat_df = extract_visual_feature_frame(image_dir, feature_set)
    feature_cols = [c for c in feat_df.columns if c != "image_name"]
    X = feat_df[feature_cols].values
    Xz = StandardScaler().fit_transform(X)

    # Same correlation-distance tree logic as voxel hierarchical clustering.
    distance_vector = clean_correlation_distance_vector(pdist(Xz, metric="correlation"))
    distance_matrix = squareform(distance_vector)
    linkage_matrix = linkage(distance_vector, method=linkage_method)

    best = choose_best_hierarchical_k(distance_vector, linkage_matrix, k_min=k_min, k_max=k_max)
    if best is None:
        raise ValueError("Could not find a valid hierarchical visual clustering.")

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(Xz)
    tsne2 = safe_tsne_precomputed(distance_matrix, random_state=random_state)
    medoid_distances = distance_to_cluster_medoids(distance_matrix, best["labels"])

    result_csv = out_dir / "visual_clusters.csv"
    features_csv = out_dir / "visual_features_only.csv"
    linkage_csv = out_dir / "visual_linkage.csv"
    dendrogram_path = out_dir / "visual_dendrogram.png"

    result_df = feat_df.copy()
    result_df["cluster_label"] = best["labels"]
    result_df["feature_set"] = feature_set
    result_df["projection_used_for_clustering"] = "correlation_distance"
    result_df["clustering_method"] = "hierarchical"
    result_df["distance_metric"] = "correlation"
    result_df["linkage_method"] = linkage_method
    result_df["plot_x"] = pca2[:, 0]
    result_df["plot_y"] = pca2[:, 1]
    result_df["pca_x"] = pca2[:, 0]
    result_df["pca_y"] = pca2[:, 1]
    result_df["tsne_x"] = tsne2[:, 0]
    result_df["tsne_y"] = tsne2[:, 1]
    result_df["distance_to_centroid"] = medoid_distances
    result_df["distance_to_medoid"] = medoid_distances
    result_df.to_csv(result_csv, index=False)
    feat_df.to_csv(features_csv, index=False)

    pd.DataFrame(
        linkage_matrix,
        columns=["child_1", "child_2", "distance", "n_observations"],
    ).to_csv(linkage_csv, index=False)

    if save_dendrogram:
        save_dendrogram_png(
            linkage_matrix,
            feat_df["image_name"].astype(str),
            dendrogram_path,
            title="Visual features hierarchical correlation dendrogram",
        )

    summary = {
        "n_images": int(len(feat_df)),
        "result_csv": artifact_path(result_csv),
        "feature_set": feature_set,
        "projection_used_for_clustering": "correlation_distance",
        "clustering_method": "hierarchical",
        "distance_metric": "correlation",
        "linkage_method": linkage_method,
        "normalization": "feature_zscore",
        "best_k": int(best["k"]),
        "silhouette": float(best["silhouette"]),
        "cluster_sizes": cluster_size_dict(best["labels"]),
        "feature_columns": feature_cols,
        "linkage_csv": artifact_path(linkage_csv),
        "dendrogram_png": artifact_path(dendrogram_path) if save_dendrogram else None,
        "distance_to_cluster_representative": "medoid_correlation_distance",
    }
    with open(out_dir / "visual_cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved hierarchical correlation visual outputs to: {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Cluster images using voxel CSVs or ImagePlot-style visual features.")
    sub = parser.add_subparsers(dest="mode", required=True)

    voxel = sub.add_parser("voxel", help="Cluster each image-by-voxel CSV separately.")
    voxel.add_argument("--input_dir", type=Path, required=True)
    voxel.add_argument("--out_dir", type=Path, required=True)
    voxel.add_argument("--pca_dim", type=int, default=50)
    voxel.add_argument("--random_state", type=int, default=42)

    voxel_hier = sub.add_parser(
        "voxel-hierarchical",
        help="Cluster each image-by-voxel CSV with correlation distance and hierarchical clustering.",
    )
    voxel_hier.add_argument("--input_dir", type=Path, required=True)
    voxel_hier.add_argument("--out_dir", type=Path, required=True)
    voxel_hier.add_argument(
        "--linkage_method",
        choices=["single", "complete", "average", "weighted"],
        default="average",
    )
    voxel_hier.add_argument("--k_min", type=int, default=2)
    voxel_hier.add_argument("--k_max", type=int, default=5)
    voxel_hier.add_argument("--random_state", type=int, default=42)
    voxel_hier.add_argument("--skip_dendrogram", action="store_true")

    visual = sub.add_parser("visual", help="Cluster images using ImagePlot-style visual features.")
    visual.add_argument("--image_dir", type=Path, required=True)
    visual.add_argument("--out_dir", type=Path, required=True)
    visual.add_argument("--feature_set", choices=["basic", "all"], default="all")
    visual.add_argument("--projection", choices=["pca", "tsne"], default="pca")
    visual.add_argument("--random_state", type=int, default=42)

    visual_hier = sub.add_parser(
        "visual-hierarchical",
        help="Cluster visual features with correlation distance and hierarchical clustering.",
    )
    visual_hier.add_argument("--image_dir", type=Path, required=True)
    visual_hier.add_argument("--out_dir", type=Path, required=True)
    visual_hier.add_argument("--feature_set", choices=["basic", "all"], default="all")
    visual_hier.add_argument(
        "--linkage_method",
        choices=["single", "complete", "average", "weighted"],
        default="average",
    )
    visual_hier.add_argument("--k_min", type=int, default=2)
    visual_hier.add_argument("--k_max", type=int, default=5)
    visual_hier.add_argument("--random_state", type=int, default=42)
    visual_hier.add_argument("--skip_dendrogram", action="store_true")

    dreamsim = sub.add_parser(
        "dreamsim",
        help="Cluster precomputed DreamSim embeddings with PCA + KMeans/euclidean.",
    )
    dreamsim.add_argument("--pth", type=Path, default=Path("data/dreamsim/dreamsim_embeddings.pth"))
    dreamsim.add_argument("--embedding_key", default="murty185")
    dreamsim.add_argument("--image_dir", type=Path, default=Path("public/images"))
    dreamsim.add_argument("--out_dir", type=Path, required=True)
    dreamsim.add_argument("--pca_dim", type=int, default=50)
    dreamsim.add_argument("--random_state", type=int, default=42)

    dreamsim_hier = sub.add_parser(
        "dreamsim-hierarchical",
        help="Cluster precomputed DreamSim embeddings with correlation distance and hierarchical clustering.",
    )
    dreamsim_hier.add_argument("--pth", type=Path, default=Path("data/dreamsim/dreamsim_embeddings.pth"))
    dreamsim_hier.add_argument("--embedding_key", default="murty185")
    dreamsim_hier.add_argument("--image_dir", type=Path, default=Path("public/images"))
    dreamsim_hier.add_argument("--out_dir", type=Path, required=True)
    dreamsim_hier.add_argument(
        "--linkage_method",
        choices=["single", "complete", "average", "weighted"],
        default="average",
    )
    dreamsim_hier.add_argument("--k_min", type=int, default=2)
    dreamsim_hier.add_argument("--k_max", type=int, default=5)
    dreamsim_hier.add_argument("--random_state", type=int, default=42)
    dreamsim_hier.add_argument("--skip_dendrogram", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "voxel":
        run_voxel_mode(args.input_dir, args.out_dir, args.pca_dim, args.random_state)
    elif args.mode == "voxel-hierarchical":
        run_voxel_hierarchical_mode(
            args.input_dir,
            args.out_dir,
            args.linkage_method,
            args.k_min,
            args.k_max,
            args.random_state,
            save_dendrogram=not args.skip_dendrogram,
        )
    elif args.mode == "visual":
        run_visual_mode(args.image_dir, args.out_dir, args.feature_set, args.projection, args.random_state)
    elif args.mode == "visual-hierarchical":
        run_visual_hierarchical_mode(
            args.image_dir,
            args.out_dir,
            args.feature_set,
            args.linkage_method,
            args.k_min,
            args.k_max,
            args.random_state,
            save_dendrogram=not args.skip_dendrogram,
        )
    elif args.mode == "dreamsim":
        cluster_dreamsim_embeddings(
            args.pth,
            args.out_dir,
            args.embedding_key,
            args.image_dir,
            args.pca_dim,
            args.random_state,
        )
    elif args.mode == "dreamsim-hierarchical":
        cluster_dreamsim_embeddings_hierarchical_corr(
            args.pth,
            args.out_dir,
            args.embedding_key,
            args.image_dir,
            args.linkage_method,
            args.k_min,
            args.k_max,
            args.random_state,
            save_dendrogram=not args.skip_dendrogram,
        )


if __name__ == "__main__":
    main()
