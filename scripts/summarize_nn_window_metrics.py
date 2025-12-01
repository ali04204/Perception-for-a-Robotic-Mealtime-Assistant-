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


LABEL_NAMES = ["not_ready", "ready"]
LABELS = [0, 1]


def summarize_split(split: str):
    metrics_dir = Path("results") / "metrics"
    csv_path = metrics_dir / f"baseline_nn_{split}_windows.csv"
    assert csv_path.exists(), f"Missing CSV for split {split}: {csv_path}"

    df = pd.read_csv(csv_path)
    y_true = df["true_idx"].to_numpy()
    y_pred = df["pred_idx"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    # classification_report with per class precision, recall, f1
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

    # Save JSON
    json_path = metrics_dir / f"baseline_nn_{split}_windows.json"
    with json_path.open("w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON for {split} to {json_path}")

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
    ax.set_title(f"NN confusion matrix ({split})")

    # Add counts on cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig_path = plots_dir / f"baseline_nn_confusion_{split}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved confusion plot for {split} to {fig_path}")

    return metrics


def main():
    all_metrics = {}
    for split in ["val", "test"]:
        print(f"\n=== {split.upper()} ===")
        metrics = summarize_split(split)
        all_metrics[split] = metrics

    # Optional combined summary
    summary_path = Path("results") / "metrics" / "baseline_nn_val_test_windows.json"
    with summary_path.open("w", encoding="utf8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved combined val test summary to {summary_path}")


if __name__ == "__main__":
    main()
