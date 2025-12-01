import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from baseline_nn import build_window_mlp
from feature_config import WINDOW_FEATURE_COLUMNS


LABEL_MAP = {"not_ready": 0, "ready": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def run_split(model, device, df_split: pd.DataFrame, split_name: str):
    """Run the NN on one split and return metrics and per window predictions."""
    if df_split.empty:
        print(f"{split_name}: no rows, skipping")
        return None, None

    X = df_split[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_true_idx = df_split["label"].map(LABEL_MAP).astype(np.int64).to_numpy()

    X_t = torch.from_numpy(X).to(device)

    model.eval()
    all_probs = []
    with torch.no_grad():
        # You can use mini batches if you want, but for this size one shot is fine
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
        all_probs = probs.cpu().numpy()

    y_pred_idx = np.argmax(all_probs, axis=1)
    y_pred_label = np.vectorize(IDX_TO_LABEL.get)(y_pred_idx)

    # Metrics
    acc = accuracy_score(y_true_idx, y_pred_idx)
    f1 = f1_score(y_true_idx, y_pred_idx, average="binary", zero_division=0)
    cm = confusion_matrix(y_true_idx, y_pred_idx).tolist()

    metrics = {
        "split": split_name,
        "acc": float(acc),
        "f1": float(f1),
        "confusion": cm,
        "num_windows": int(len(df_split)),
    }

    # Build output dataframe with any id columns that exist
    base_cols = []
    for c in ["clip_id", "video_id", "person_id", "segment_id", "window_start", "window_end"]:
        if c in df_split.columns:
            base_cols.append(c)

    out_df = pd.DataFrame()
    if base_cols:
        out_df[base_cols] = df_split[base_cols]

    out_df["split"] = split_name
    out_df["true_label"] = df_split["label"].values
    out_df["true_idx"] = y_true_idx
    out_df["pred_idx"] = y_pred_idx
    out_df["pred_label"] = y_pred_label
    out_df["prob_not_ready"] = all_probs[:, 0]
    out_df["prob_ready"] = all_probs[:, 1]

    return metrics, out_df


def main():
    # Load config to get input dim and file names
    cfg_path = Path("results/config/baseline_nn_windows.json")
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    with cfg_path.open("r", encoding="utf8") as f:
        cfg = json.load(f)

    in_features = cfg["input_dim"]

    # Load all_windows
    all_windows_path = Path("results/features/all_windows.csv")
    assert all_windows_path.exists(), f"Missing all_windows: {all_windows_path}"
    ds = pd.read_csv(all_windows_path)

    # Build model and load best weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_window_mlp(in_features=in_features)
    model.to(device)

    model_path = Path("results/models/baseline_nn.pt")
    assert model_path.exists(), f"Missing model weights: {model_path}"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    out_dir = Path("results") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for split_name in ["val", "test"]:
        df_split = ds[ds["split"] == split_name].copy()
        print(f"{split_name}: {len(df_split)} windows")

        metrics, out_df = run_split(model, device, df_split, split_name)
        if metrics is None:
            continue

        all_metrics[split_name] = metrics

        csv_path = out_dir / f"baseline_nn_{split_name}_windows.csv"
        out_df.to_csv(csv_path, index=False)
        print(f"Saved per window predictions for {split_name} to {csv_path}")

    # Save a small metrics summary json for quick reference
    summary_path = out_dir / "baseline_nn_val_test_window_metrics.json"
    with summary_path.open("w", encoding="utf8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved val and test window metrics to {summary_path}")


if __name__ == "__main__":
    main()
