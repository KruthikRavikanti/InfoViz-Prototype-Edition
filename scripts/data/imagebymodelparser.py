from pathlib import Path
import pandas as pd
import re
import argparse

ROI_ORDER = ["ffa", "ppa", "eba"]

FILENAME_RE = re.compile(
    r"^murtylab_(?P<model>.+?)_nsd_1000_(?P<roi>ffa|ppa|eba)_(?P<timestamp>\d+Z)\.csv$"
)

ID_CANDIDATES = [
    "image_name",
    "image",
    "image_id",
    "img",
    "img_id",
    "stimulus",
    "stimulus_id",
]


def detect_id_column(df: pd.DataFrame):
    for col in df.columns:
        if col.lower() in ID_CANDIDATES:
            return col
    return df.columns[0]  # fallback: use first column


def parse_files(input_dir: Path):
    files_by_key = {}

    for path in input_dir.glob("*.csv"):
        match = FILENAME_RE.match(path.name)
        if not match:
            print(f"Skipping unmatched filename: {path.name}")
            continue

        model = match.group("model")
        roi = match.group("roi")
        timestamp = match.group("timestamp")
        key = (model, roi)

        if key not in files_by_key or timestamp > files_by_key[key][0]:
            files_by_key[key] = (timestamp, path)

    parsed = []
    for (model, roi), (_, path) in files_by_key.items():
        parsed.append((model, roi, path))

    parsed.sort(key=lambda x: (x[0], ROI_ORDER.index(x[1])))
    return parsed


def reduce_file_to_mean(path: Path):
    df = pd.read_csv(path)

    id_col = detect_id_column(df)
    image_names = df[id_col].copy()

    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    drop_cols.append(id_col)

    voxel_df = df.drop(columns=drop_cols, errors="ignore")
    voxel_df = voxel_df.apply(pd.to_numeric, errors="coerce")

    if voxel_df.shape[1] == 0:
        raise ValueError(f"No numeric voxel columns found in {path.name}")

    row_mean = voxel_df.mean(axis=1)
    return image_names, row_mean


def build_combined_csv(input_dir: str, output_file: str):
    input_path = Path(input_dir)
    parsed_files = parse_files(input_path)

    if not parsed_files:
        raise ValueError("No matching CSV files found.")

    combined = None
    discovered_models = set()

    for model, roi, path in parsed_files:
        col_name = f"{model}_{roi}"
        discovered_models.add(model)

        image_names, row_mean = reduce_file_to_mean(path)

        current_df = pd.DataFrame({
            "image_name": image_names,
            col_name: row_mean
        })

        if combined is None:
            combined = current_df
        else:
            combined = combined.merge(current_df, on="image_name", how="inner")

    ordered_feature_cols = []
    for model in sorted(discovered_models):
        for roi in ROI_ORDER:
            col_name = f"{model}_{roi}"
            if col_name in combined.columns:
                ordered_feature_cols.append(col_name)

    combined = combined[["image_name"] + ordered_feature_cols]
    combined.to_csv(output_file, index=False)

    print(f"Saved to: {output_file}")
    print(f"Shape: {combined.shape}")
    print(combined.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="combined_image_means.csv")
    args = parser.parse_args()

    build_combined_csv(args.input_dir, args.output_file)