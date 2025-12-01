import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from feature_config import WINDOW_FEATURE_COLUMNS

LABEL_MAP = {"not_ready": 0, "ready": 1}


def build_sequences(ds: pd.DataFrame, split: str, group_col: str, scaler):
    df = ds[ds["split"] == split].copy()
    if df.empty:
        print(f"{split}: no rows")
        return [], [], []

    print(f"{split}: {len(df)} windows before grouping")

    seq_X = []
    seq_y = []
    clip_ids = []

    # sort inside each clip by time if we have window_start
    has_time = "window_start" in df.columns

    for cid, g in df.groupby(group_col):
        if has_time:
            g = g.sort_values("window_start")
        # features
        X_raw = g[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        # labels
        y = g["label"].map(LABEL_MAP).astype(np.int64).to_numpy()

        seq_X.append(X_scaled)
        seq_y.append(y)
        clip_ids.append(cid)

    print(f"{split}: {len(seq_X)} sequences")
    return seq_X, seq_y, clip_ids


def pad_sequences(seq_X, seq_y):
    """Pad sequences to the same length with zeros and -1 labels."""
    if not seq_X:
        return (
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    lengths = np.array([len(s) for s in seq_X], dtype=np.int64)
    max_len = int(lengths.max())
    feat_dim = seq_X[0].shape[1]

    X_padded = np.zeros((len(seq_X), max_len, feat_dim), dtype=np.float32)
    y_padded = -1 * np.ones((len(seq_y), max_len), dtype=np.int64)

    for i, (x, y) in enumerate(zip(seq_X, seq_y)):
        L = len(x)
        X_padded[i, :L, :] = x
        y_padded[i, :L] = y

    return X_padded, y_padded, lengths


def main():
    all_windows_path = Path("results/features/all_windows.csv")
    scaler_path = Path("results/config/seq_nn_window_scaler.joblib")

    assert all_windows_path.exists(), f"Missing {all_windows_path}"
    assert scaler_path.exists(), f"Missing scaler {scaler_path}"

    ds = pd.read_csv(all_windows_path)
    scaler = load(scaler_path)

    # choose grouping column
    group_col = None
    for cand in ["clip_id", "video_id"]:
        if cand in ds.columns:
            group_col = cand
            break
    assert group_col is not None, "Need clip_id or video_id column for sequence grouping"
    print(f"Grouping sequences by {group_col}")

    splits = {}
    meta = {"group_col": group_col, "label_map": LABEL_MAP}

    for split in ["train", "val", "test"]:
        seq_X, seq_y, clip_ids = build_sequences(ds, split, group_col, scaler)
        X_pad, y_pad, lengths = pad_sequences(seq_X, seq_y)

        print(
            f"{split}: X shape {X_pad.shape}, y shape {y_pad.shape}, num_seqs={len(clip_ids)}, max_len={lengths.max() if len(lengths) > 0 else 0}"
        )

        splits[split] = {
            "X": X_pad,
            "y": y_pad,
            "lengths": lengths,
            "clip_ids": np.array(clip_ids),
        }

    out_dir = Path("results/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "seq_nn_datasets.npz"
    np.savez(
        out_npz,
        X_train=splits["train"]["X"],
        y_train=splits["train"]["y"],
        len_train=splits["train"]["lengths"],
        clips_train=splits["train"]["clip_ids"],
        X_val=splits["val"]["X"],
        y_val=splits["val"]["y"],
        len_val=splits["val"]["lengths"],
        clips_val=splits["val"]["clip_ids"],
        X_test=splits["test"]["X"],
        y_test=splits["test"]["y"],
        len_test=splits["test"]["lengths"],
        clips_test=splits["test"]["clip_ids"],
    )
    print(f"Saved sequence NN datasets to {out_npz}")

    # small meta json
    meta_path = Path("results/config/seq_nn_meta.json")
    with meta_path.open("w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved sequence NN meta to {meta_path}")


if __name__ == "__main__":
    main()
