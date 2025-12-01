import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

LABEL_MAP = {"not_ready": 0, "ready": 1}
LABEL_NAMES = ["not_ready", "ready"]
LABELS = [0, 1]


def summarize_split(ds: pd.DataFrame, split: str):
    df = ds[(ds["split"] == split) & ds["nn_smoothed_label"].notna()].copy()
    if df.empty:
        print(f"{split}: no smoothed rows, skipping")
        return None

    y_true = df["label"].map(LABEL_MAP).to_numpy()
    y_pred = df["nn_smoothed_label"].map(LABEL_MAP).to_numpy()

    acc = accuracy_score(y_true, y_pred)
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

    metrics_dir = Path("results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"baseline_nn_{split}_windows_smoothed.json"
    with json_path.open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved smoothed metrics JSON for {split} to {json_path}")

    # Plot confusion matrix
    plots_dir = Path("results") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(LABEL_NAMES)))
    ax.set_yticks(np.arange(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"NN smoothed confusion ({split})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig_path = plots_dir / f"baseline_nn_confusion_{split}_smoothed.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved smoothed confusion plot for {split} to {fig_path}")

    return metrics


def main():
    all_windows_path = Path("results/features/all_windows_nn_smoothed.csv")
    assert all_windows_path.exists(), f"Missing smoothed windows file: {all_windows_path}"

    ds = pd.read_csv(all_windows_path)

    all_metrics = {}
    for split in ["val", "test"]:
        print(f"\n=== {split.upper()} (smoothed) ===")
        m = summarize_split(ds, split)
        if m is not None:
            all_metrics[split] = m

    summary_path = Path("results/metrics/baseline_nn_val_test_windows_smoothed.json")
    with summary_path.open("w", encoding="utf8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved combined smoothed val test summary to {summary_path}")


if __name__ == "__main__":
    main()
