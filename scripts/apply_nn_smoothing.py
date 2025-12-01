import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline_nn import build_window_mlp
from feature_config import WINDOW_FEATURE_COLUMNS

LABEL_MAP = {"not_ready": 0, "ready": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def smooth_sequence(p_ready, enter_th, exit_th, min_enter, min_exit):
    """Simple hysteresis smoothing over a 1d probability sequence."""
    state = 0  # 0 = not_ready, 1 = ready
    enter_count = 0
    exit_count = 0
    smoothed = []

    for p in p_ready:
        if state == 0:
            if p >= enter_th:
                enter_count += 1
            else:
                enter_count = 0

            if enter_count >= min_enter:
                state = 1
                enter_count = 0
                exit_count = 0
        else:
            if p <= exit_th:
                exit_count += 1
            else:
                exit_count = 0

            if exit_count >= min_exit:
                state = 0
                exit_count = 0
                enter_count = 0

        smoothed.append(state)

    return smoothed


def main():
    # Load configs
    nn_cfg_path = Path("results/config/baseline_nn_windows.json")
    smooth_cfg_path = Path("results/config/smoothing_nn.json")
    all_windows_path = Path("results/features/all_windows.csv")
    model_path = Path("results/models/baseline_nn.pt")

    assert nn_cfg_path.exists(), f"Missing NN config: {nn_cfg_path}"
    assert smooth_cfg_path.exists(), f"Missing smoothing config: {smooth_cfg_path}"
    assert all_windows_path.exists(), f"Missing all_windows: {all_windows_path}"
    assert model_path.exists(), f"Missing NN model weights: {model_path}"

    with nn_cfg_path.open("r", encoding="utf8") as f:
        nn_cfg = json.load(f)

    with smooth_cfg_path.open("r", encoding="utf8") as f:
        smooth_cfg = json.load(f)

    enter_th = smooth_cfg["enter_threshold"]
    exit_th = smooth_cfg["exit_threshold"]
    min_enter = smooth_cfg["min_windows_enter"]
    min_exit = smooth_cfg["min_windows_exit"]

    in_features = nn_cfg["input_dim"]

    # Load data
    ds = pd.read_csv(all_windows_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build and load model
    model = build_window_mlp(in_features=in_features)
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Compute nn_p_ready for all windows
    X = ds[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    nn_p_ready = probs[:, 1]
    ds["nn_p_ready"] = nn_p_ready

    # Prepare smoothed label column (only for val and test)
    ds["nn_smoothed_label"] = pd.Series([None] * len(ds), dtype="object")

    # Choose clip grouping column
    group_col = None
    for cand in ["clip_id", "video_id"]:
        if cand in ds.columns:
            group_col = cand
            break

    if group_col is None:
        print("Warning: no clip_id or video_id column found, smoothing over whole val and test only")
        group_keys = [None]
    else:
        group_keys = ds[group_col].unique()

    # Sort key inside each clip
    sort_col = "window_start" if "window_start" in ds.columns else None

    # Apply smoothing for val and test splits
    for split in ["val", "test"]:
        df_split_idx = ds["split"] == split
        if not df_split_idx.any():
            print(f"{split}: no rows, skipping")
            continue

        print(f"Smoothing {split} windows")

        if group_col is None:
            # Treat entire split as one sequence
            idx = df_split_idx
            if sort_col is not None:
                ds_split = ds.loc[idx].sort_values(sort_col)
            else:
                ds_split = ds.loc[idx]

            p_seq = ds_split["nn_p_ready"].to_numpy()
            smoothed = smooth_sequence(p_seq, enter_th, exit_th, min_enter, min_exit)
            smoothed = np.array(smoothed, dtype=int)

            # Map back into main dataframe
            ds.loc[ds_split.index, "nn_smoothed_label"] = [
                IDX_TO_LABEL[int(s)] for s in smoothed
            ]
        else:
            # Group by clip inside this split
            for key in group_keys:
                # subset rows for this split and this clip
                mask = df_split_idx & (ds[group_col] == key)
                if not mask.any():
                    continue

                if sort_col is not None:
                    ds_clip = ds.loc[mask].sort_values(sort_col)
                else:
                    ds_clip = ds.loc[mask]

                p_seq = ds_clip["nn_p_ready"].to_numpy()
                smoothed = smooth_sequence(p_seq, enter_th, exit_th, min_enter, min_exit)
                smoothed = np.array(smoothed, dtype=int)

                ds.loc[ds_clip.index, "nn_smoothed_label"] = [
                    IDX_TO_LABEL[int(s)] for s in smoothed
                ]

    out_path = Path("results/features/all_windows_nn_smoothed.csv")
    ds.to_csv(out_path, index=False)
    print(f"Wrote smoothed NN windows to {out_path}")


if __name__ == "__main__":
    main()
