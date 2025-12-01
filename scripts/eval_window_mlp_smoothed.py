import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from feature_config import WINDOW_FEATURE_COLUMNS



def smooth_clip_predictions(df_clip: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Smooth p_ready per clip using a rolling mean over p_ready_raw.

    We sort inside a clip if we have a time column. If not, we keep the
    original row order.
    """
    df_clip = df_clip.copy()

    if "t0_sec" in df_clip.columns:
        df_clip = df_clip.sort_values("t0_sec")
    elif "t_start" in df_clip.columns:
        df_clip = df_clip.sort_values("t_start")
    else:
        df_clip = df_clip.sort_index()

    smoothed = (
        df_clip["p_ready_raw"]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )

    df_clip["p_ready_smooth"] = smoothed
    df_clip["y_pred_smooth"] = (smoothed >= 0.5).astype(int)
    return df_clip


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate window MLP NN on test split with temporal smoothing."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to a saved PyTorch model, e.g. results/models/window_mlp_best.pt",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Size of rolling window for smoothing over p_ready.",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]

    windows_path = repo_root / "results" / "features" / "all_windows.csv"
    metrics_path = repo_root / "results" / "metrics" / "window_mlp_test_windows_smoothed.json"
    preds_path = repo_root / "results" / "predictions" / "window_mlp_test_windows_smoothed.csv"

    assert windows_path.exists(), f"Missing dataset csv: {windows_path}"

    ds = pd.read_csv(windows_path)

    if "split" not in ds.columns:
        raise ValueError("Dataset is missing 'split' column in all_windows.csv")

    ds_test = ds[ds["split"] == "test"].copy()
    if ds_test.empty:
        raise ValueError("No test rows found in all_windows.csv. Check your splits.")

    if "label" not in ds_test.columns:
        raise ValueError("Dataset is missing 'label' column in all_windows.csv")

    # Label encoding
    ds_test["label_int"] = ds_test["label"].map({"not_ready": 0, "ready": 1}).astype(int)

    # Feature matrix
    missing = [c for c in WINDOW_FEATURE_COLUMNS if c not in ds_test.columns]
    if missing:
        msg = "Dataset is missing expected feature columns:\n" + "\n".join("  " + c for c in missing)
        raise ValueError(msg)

    X_test = ds_test[WINDOW_FEATURE_COLUMNS].to_numpy().astype(np.float32)
    y_true = ds_test["label_int"].to_numpy()

    # Identify clip id column
    if "clip_id" in ds_test.columns:
        id_col = "clip_id"
    elif "video_id" in ds_test.columns:
        id_col = "video_id"
    else:
        raise ValueError("Expected 'clip_id' or 'video_id' column in all_windows.csv")

    # Load model state_dict and rebuild the MLP
    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model state_dict from {checkpoint_path}")
    device = torch.device("cpu")

    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict (dict), got {type(state_dict)}")

    # Infer layer sizes from the state_dict
    w0 = state_dict["net.0.weight"]
    h1, in_dim = w0.shape

    w4 = state_dict["net.4.weight"]
    h2, h1_check = w4.shape
    assert h1_check == h1, f"Mismatch between net.0 and net.4 shapes: {h1_check} vs {h1}"

    w8 = state_dict["net.8.weight"]
    out_dim, h2_check = w8.shape
    assert h2_check == h2, f"Mismatch between net.4 and net.8 shapes: {h2_check} vs {h2}"

    # Sanity check against feature config
    expected_in_dim = len(WINDOW_FEATURE_COLUMNS)
    if in_dim != expected_in_dim:
        raise ValueError(
            f"Input dim from state_dict ({in_dim}) does not match number of "
            f"WINDOW_FEATURE_COLUMNS ({expected_in_dim})"
        )

    class WindowMLP(nn.Module):
        def __init__(self, in_dim, h1, h2, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1),      # net.0
                nn.ReLU(),                  # net.1 (no params)
                nn.BatchNorm1d(h1),         # net.2
                nn.ReLU(),                  # net.3
                nn.Linear(h1, h2),          # net.4
                nn.ReLU(),                  # net.5
                nn.BatchNorm1d(h2),         # net.6
                nn.ReLU(),                  # net.7
                nn.Linear(h2, out_dim),     # net.8
            )

        def forward(self, x):
            return self.net(x)

    model = WindowMLP(in_dim, h1, h2, out_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()


    # Forward pass
    with torch.no_grad():
        # simple batch to avoid memory spikes
        batch_size = 512
        probs = []
        for i in range(0, X_test.shape[0], batch_size):
            xb = torch.from_numpy(X_test[i : i + batch_size]).to(device)
            logits = model(xb)
            if logits.shape[-1] == 1:
                # binary classifier with a single logit
                pb = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            else:
                # multiclass with 2 outputs
                pb = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(pb)
        p_ready_raw = np.concatenate(probs, axis=0)

    y_pred_raw = (p_ready_raw >= 0.5).astype(int)

    ds_test["p_ready_raw"] = p_ready_raw
    ds_test["y_pred_raw"] = y_pred_raw

    # Ensure id column is present and string typed
    ds_test[id_col] = ds_test[id_col].astype(str)

    # Smooth per clip
    smoothed_clips = []
    for cid, df_clip in ds_test.groupby(id_col):
        df_smooth = smooth_clip_predictions(df_clip, window_size=args.window_size)
        smoothed_clips.append(df_smooth)

    ds_smooth = pd.concat(smoothed_clips, axis=0)

    # Stable order
    sort_cols = [c for c in [id_col, "t0_sec"] if c in ds_smooth.columns]
    if sort_cols:
        ds_smooth = ds_smooth.sort_values(sort_cols)
    else:
        ds_smooth = ds_smooth.sort_index()

    ds_smooth = ds_smooth.reset_index(drop=True)

    # Standardize id column name for downstream scripts
    if id_col != "clip_id":
        ds_smooth = ds_smooth.rename(columns={id_col: "clip_id"})
        id_col = "clip_id"

    # Metrics
    y_true_final = ds_smooth["label_int"].to_numpy()
    y_pred_smooth = ds_smooth["y_pred_smooth"].to_numpy()

    test_acc = float(accuracy_score(y_true_final, y_pred_smooth))
    test_f1_pos = float(f1_score(y_true_final, y_pred_smooth))

    test_confusion = confusion_matrix(y_true_final, y_pred_smooth).tolist()

    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true_final,
        y_pred_smooth,
        labels=[0, 1],
        zero_division=0,
    )
    f1_not_ready = float(f1_per_class[0])
    f1_ready = float(f1_per_class[1])

    metrics = {
        "test_acc": test_acc,
        "test_f1": test_f1_pos,
        "test_confusion": test_confusion,
        "accuracy": test_acc,
        "f1_not_ready": f1_not_ready,
        "f1_ready": f1_ready,
        "confusion": test_confusion,
        "window_size": args.window_size,
        "note": "Window MLP NN on test split with sliding window smoothing over p_ready.",
        "checkpoint": str(checkpoint_path),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ds_smooth.to_csv(preds_path, index=False)

    print("Wrote smoothed NN metrics to", metrics_path)
    print("Wrote smoothed NN test predictions to", preds_path)
    print("accuracy:", test_acc)
    print("f1_not_ready:", f1_not_ready)
    print("f1_ready:", f1_ready)
    print("confusion:", test_confusion)


if __name__ == "__main__":
    main()
