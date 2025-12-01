import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_MAP = {"not_ready": 0, "ready": 1}
LABEL_NAMES = ["not_ready", "ready"]
LABELS = [0, 1]


def load_split_csv(metrics_dir: Path, split: str) -> pd.DataFrame:
    csv_path = metrics_dir / f"baseline_rf_{split}_windows.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing RF per window CSV for split '{split}': {csv_path}\n"
            "Run: python scripts/eval_rf_from_windows.py results/features/all_windows.csv"
        )
    return pd.read_csv(csv_path)


def summarize_split(df: pd.DataFrame, split: str):
    # Try to use numeric columns if they exist, otherwise map from labels
    if "true_idx" in df.columns and "pred_idx" in df.columns:
        y_true = df["true_idx"].to_numpy()
        y_pred = df["pred_idx"].to_numpy()
    else:
        y_true = df["true_label"].map(LABEL_MAP).to_numpy()
        if "pred_label" in df.columns:
            y_pred = df["pred_label"].map(LABEL_MAP).to_numpy()
        else:
            raise ValueError(
                f"Cannot find true_idx/pred_idx or label columns for split {split}"
            )

    acc = accuracy_score(y_true, y_pred)
    # F1 with ready (1) as positive, to match NN
    f1_ready = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "split": split,
        "num_windows": int(len(df)),
        "accuracy": float(acc),
        "f1_ready_binary": float(f1_ready),
        "confusion": cm.tolist(),
        "per_class": {
            "not_ready": {
                "precision": report["not_ready"]["precision"],
                "recall": report["not_ready"]["recall"],
                "f1": report["not_ready"]["f1-score"],
                "support": int(report["not_ready"]["support"]),
            },
            "ready": {
                "precision": report["ready"]["precision"],
                "recall": report["ready"]["recall"],
                "f1": report["ready"]["f1-score"],
                "support": int(report["ready"]["support"]),
            },
        },
        "macro_avg": {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
        },
        "weighted_avg": {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        },
    }

    return metrics


def main():
    metrics_dir = Path("results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load RF per window predictions (these come from eval_rf_from_windows.py)
    val_df = load_split_csv(metrics_dir, "val")
    test_df = load_split_csv(metrics_dir, "test")

    val_metrics = summarize_split(val_df, "val")
    test_metrics = summarize_split(test_df, "test")

    # Save per split JSONs
    val_json_path = metrics_dir / "baseline_rf_val_windows.json"
    test_json_path = metrics_dir / "baseline_rf_test_windows.json"

    with val_json_path.open("w", encoding="utf8") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Saved RF val metrics to {val_json_path}")

    with test_json_path.open("w", encoding="utf8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved RF test metrics to {test_json_path}")

    # Combined val+test summary
    combined = {"val": val_metrics, "test": test_metrics}
    combined_path = metrics_dir / "baseline_rf_val_test_windows.json"
    with combined_path.open("w", encoding="utf8") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved combined RF val+test metrics to {combined_path}")

    # Simple top level metrics file expected by compare_rf_vs_nn.py
    top_path = metrics_dir / "baseline_rf_metrics.json"
    summary = {
        "model_type": "RandomForest window baseline",
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_ready_binary"],
        "test_confusion": test_metrics["confusion"],
    }
    with top_path.open("w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved top level RF metrics to {top_path}")


if __name__ == "__main__":
    main()
