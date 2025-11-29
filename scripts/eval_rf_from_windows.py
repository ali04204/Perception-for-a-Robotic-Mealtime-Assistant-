import json
import pathlib

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from feature_config import WINDOW_FEATURE_COLUMNS


ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOWS_CSV = ROOT / "results" / "features" / "all_windows.csv"
MODEL_PATH = ROOT / "results" / "models" / "baseline_rf.joblib"
METRICS_DIR = ROOT / "results" / "metrics"


def evaluate_split(df: pd.DataFrame, split_name: str, label_map: dict[int, str]):
    """Evaluate RF on one split and return metrics dict."""
    print(f"\n=== Evaluating split {split_name} ===")

    split_df = df[df["split"] == split_name].copy()
    if split_df.empty:
        print(f"No rows with split == {split_name}, skipping")
        return None

    # Labels
    y_true = split_df["label"].map({"not_ready": 0, "ready": 1}).astype(int).values

    # Features
    missing = [c for c in WINDOW_FEATURE_COLUMNS if c not in split_df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing feature columns for split {split_name}: {missing}"
        )

    X = split_df[WINDOW_FEATURE_COLUMNS].values

    # Predict
    y_proba = rf.predict_proba(X)
    y_pred = rf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Per class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # Detailed classification report as dict
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[label_map[0], label_map[1]],
        output_dict=True,
        zero_division=0,
    )

    # Build metrics dict for JSON
    metrics = {
        "split": split_name,
        "accuracy": float(acc),
        "labels": [label_map[0], label_map[1]],
        "confusion_matrix": cm.tolist(),
        "per_class": {
            label_map[0]: {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            label_map[1]: {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "classification_report": cls_report,
    }

    return metrics, split_df, y_proba[:, 1]


if __name__ == "__main__":
    assert WINDOWS_CSV.exists(), f"Missing {WINDOWS_CSV}"
    assert MODEL_PATH.exists(), f"Missing RF model at {MODEL_PATH}"

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading windows dataset from: {WINDOWS_CSV}")
    df = pd.read_csv(WINDOWS_CSV)

    for col in ["split", "label"]:
        assert col in df.columns, f"Expected column {col} in all_windows.csv"

    print(f"Loading RF model from: {MODEL_PATH}")
    rf = load(MODEL_PATH)

    label_map_int_to_str = {0: "not_ready", 1: "ready"}

    all_metrics = {}

    # Evaluate val and test splits
    for split_name in ["val", "test"]:
        result = evaluate_split(df, split_name, label_map_int_to_str)
        if result is None:
            continue

        metrics, split_df, p_ready = result
        all_metrics[split_name] = metrics

        # Save metrics per split
        out_path = METRICS_DIR / f"baseline_rf_{split_name}_windows.json"
        with out_path.open("w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out_path}")

    # Optionally print short summary
    print("\n=== Summary ===")
    for split_name, metrics in all_metrics.items():
        print(
            f"{split_name}: acc={metrics['accuracy']:.3f}, "
            f"F1 not_ready={metrics['per_class']['not_ready']['f1']:.3f}, "
            f"F1 ready={metrics['per_class']['ready']['f1']:.3f}"
        )
