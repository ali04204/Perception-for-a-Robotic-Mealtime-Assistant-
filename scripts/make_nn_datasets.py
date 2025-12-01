import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from feature_config import WINDOW_FEATURE_COLUMNS


LABEL_MAP = {"not_ready": 0, "ready": 1}


def build_split_arrays(df: pd.DataFrame):
    """Return X, y for a given split DataFrame."""
    if df.empty:
        # Handle the edge case so np.savez still works
        num_features = len(WINDOW_FEATURE_COLUMNS)
        X = np.zeros((0, num_features), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y

    # Features
    X = df[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Labels 0 or 1
    y = df["label"].map(LABEL_MAP).astype(np.int64).to_numpy()

    return X, y


def compute_class_weights(df: pd.DataFrame):
    """Compute simple inverse frequency class weights."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)

    weights = {}
    for name, idx in LABEL_MAP.items():
        count = counts.get(name, 0)
        if count == 0:
            # Avoid divide by zero
            weights[str(idx)] = None
        else:
            # Simple inverse frequency
            weights[str(idx)] = float(total / count)

    info = {
        "label_map": LABEL_MAP,
        "counts": counts,
        "total_windows": int(total),
        "class_weights": weights,
    }
    return info


def main(all_windows_path: pathlib.Path):
    assert all_windows_path.exists(), f"Missing file: {all_windows_path}"

    ds = pd.read_csv(all_windows_path)
    required_cols = {"split", "label"}
    missing = required_cols - set(ds.columns)
    if missing:
        raise ValueError(f"all_windows.csv is missing columns: {missing}")

    # Build splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        df_split = ds[ds["split"] == split_name].copy()
        X, y = build_split_arrays(df_split)
        splits[split_name] = {"X": X, "y": y}
        print(
            f"{split_name}: {len(df_split)} windows, "
            f"shape X = {X.shape}, y = {y.shape}"
        )

    # Save arrays to a single NPZ
    out_dir = pathlib.Path("results/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "nn_windows_datasets.npz"

    np.savez(
        npz_path,
        X_train=splits["train"]["X"],
        y_train=splits["train"]["y"],
        X_val=splits["val"]["X"],
        y_val=splits["val"]["y"],
        X_test=splits["test"]["X"],
        y_test=splits["test"]["y"],
    )
    print(f"Saved NN datasets to {npz_path}")

    # Class weights from all windows
    class_info = compute_class_weights(ds)

    config_dir = pathlib.Path("results/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config_dir / "baseline_nn_class_weights.json"
    with weights_path.open("w", encoding="utf8") as f:
        json.dump(class_info, f, indent=2)
    print(f"Saved class weights info to {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "all_windows",
        nargs="?",
        default="results/features/all_windows.csv",
        help="Path to all_windows.csv",
    )
    args = parser.parse_args()
    main(pathlib.Path(args.all_windows))
