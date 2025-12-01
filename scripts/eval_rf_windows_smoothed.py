import json
import pathlib

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Adjust this import if your feature config lives in configs/
# from configs.feature_config import WINDOW_FEATURE_COLUMNS
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
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    windows_path = repo_root / "results" / "features" / "all_windows.csv"
    model_path = repo_root / "results" / "models" / "baseline_rf.joblib"
    metrics_path = repo_root / "results" / "metrics" / "baseline_rf_test_windows_smoothed.json"
    preds_path = repo_root / "results" / "predictions" / "baseline_rf_test_windows_smoothed.csv"

    assert windows_path.exists(), f"Missing dataset csv: {windows_path}"
    assert model_path.exists(), f"Missing RF model: {model_path}"

    ds = pd.read_csv(windows_path)

    # 1) Filter to test split
    if "split" not in ds.columns:
        raise ValueError("Dataset is missing 'split' column. Make sure Phase 3 merges are done.")

    ds_test = ds[ds["split"] == "test"].copy()
    if ds_test.empty:
        raise ValueError("No test rows found in all_windows.csv. Check your splits.")

    if "label" not in ds_test.columns:
        raise ValueError("Dataset is missing 'label' column.")

    # 2) Feature matrix
    missing = [c for c in WINDOW_FEATURE_COLUMNS if c not in ds_test.columns]
    if missing:
        msg = "Dataset is missing expected feature columns:\n" + "\n".join("  " + c for c in missing)
        raise ValueError(msg)

    X_test = ds_test[WINDOW_FEATURE_COLUMNS].to_numpy()

    # 3) Load RF and get raw probabilities
    rf = load(model_path)
    proba = rf.predict_proba(X_test)
    p_ready_raw = proba[:, 1]
    y_pred_raw = (p_ready_raw >= 0.5).astype(int)

    ds_test["p_ready_raw"] = p_ready_raw
    ds_test["y_pred_raw"] = y_pred_raw

    # 4) Make sure we have a clip id to group by
    if "clip_id" not in ds_test.columns and "video_id" in ds_test.columns:
        ds_test = ds_test.rename(columns={"video_id": "clip_id"})

    if "clip_id" not in ds_test.columns:
        raise ValueError("Expected a 'clip_id' or 'video_id' column to group windows per clip.")

    # 5) Smooth per clip
    smoothed_clips = []
    for clip_id, df_clip in ds_test.groupby("clip_id"):
        df_smooth = smooth_clip_predictions(df_clip, window_size=5)
        smoothed_clips.append(df_smooth)

    ds_smooth = pd.concat(smoothed_clips, axis=0)

    # Keep a stable global order for saving
    sort_cols = [c for c in ["clip_id", "t0_sec"] if c in ds_smooth.columns]
    if sort_cols:
        ds_smooth = ds_smooth.sort_values(sort_cols)
    else:
        ds_smooth = ds_smooth.sort_index()

    ds_smooth = ds_smooth.reset_index(drop=True)

    # 6) Metrics: true labels aligned with ds_smooth
    y_true = ds_smooth["label"].map({"not_ready": 0, "ready": 1}).astype(int).to_numpy()
    y_pred_smooth = ds_smooth["y_pred_smooth"].to_numpy()

    # Overall accuracy and positive-class F1 (for consistency with unsmoothed RF)
    test_acc = float(accuracy_score(y_true, y_pred_smooth))
    # F1 for positive class (ready == 1)
    test_f1 = float(f1_score(y_true, y_pred_smooth))

    # Confusion matrix
    test_confusion = confusion_matrix(y_true, y_pred_smooth).tolist()

    # Per-class F1 (0 = not_ready, 1 = ready)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true,
        y_pred_smooth,
        labels=[0, 1],
        zero_division=0,
    )
    f1_not_ready = float(f1_per_class[0])
    f1_ready = float(f1_per_class[1])

    # 7) Build metrics dict with keys that compare_rf_vs_nn.py expects
    metrics = {
        # old style keys (unsmoothed RF style)
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_confusion": test_confusion,
        # new generic keys for smoothed comparison script
        "accuracy": test_acc,
        "f1_not_ready": f1_not_ready,
        "f1_ready": f1_ready,
        "confusion": test_confusion,
        # extra info
        "window_size": 5,
        "note": "RF baseline on test split with sliding window smoothing over p_ready.",
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ds_smooth.to_csv(preds_path, index=False)

    print("Wrote smoothed RF metrics to", metrics_path)
    print("Wrote smoothed RF test predictions to", preds_path)
    print("test_acc:", test_acc)
    print("test_f1 (ready class):", test_f1)
    print("f1_not_ready:", f1_not_ready)
    print("f1_ready:", f1_ready)
    print("test_confusion:", test_confusion)


if __name__ == "__main__":
    main()
