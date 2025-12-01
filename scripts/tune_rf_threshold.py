import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


LABEL_MAP = {"not_ready": 0, "ready": 1}
LABELS = [0, 1]


def load_split(metrics_dir: Path, split: str) -> pd.DataFrame:
    path = metrics_dir / f"baseline_rf_{split}_windows.csv"
    assert path.exists(), f"Missing RF per window CSV for {split}: {path}"
    return pd.read_csv(path)


def apply_threshold(df: pd.DataFrame, thresh: float):
    y_true = df["true_label"].map(LABEL_MAP).to_numpy()
    p_ready = df["prob_ready"].to_numpy()
    y_pred = (p_ready >= thresh).astype(int)
    return y_true, y_pred


def metrics_for(df: pd.DataFrame, thresh: float, name: str):
    y_true, y_pred = apply_threshold(df, thresh)
    acc = accuracy_score(y_true, y_pred)
    f1_ready = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    print(f"{name}: thresh={thresh:.3f} acc={acc:.3f} f1_ready={f1_ready:.3f} cm={cm}")
    return {
        "threshold": float(thresh),
        "accuracy": float(acc),
        "f1_ready": float(f1_ready),
        "confusion": cm,
    }


def main():
    metrics_dir = Path("results") / "metrics"

    df_val = load_split(metrics_dir, "val")
    df_test = load_split(metrics_dir, "test")

    # Sweep thresholds on val from 0.2 to 0.8
    best_t = None
    best_f1 = -1.0
    history = []

    print("Sweeping thresholds on VAL")
    for t in np.linspace(0.2, 0.8, 25):
        y_true, y_pred = apply_threshold(df_val, t)
        f1_ready = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        history.append({"threshold": float(t), "f1_ready": float(f1_ready)})
        print(f"  t={t:.3f} val_f1_ready={f1_ready:.3f}")
        if f1_ready > best_f1:
            best_f1 = f1_ready
            best_t = t

    print("\nBest threshold on VAL:")
    print(f"  best_t={best_t:.3f} best_val_f1_ready={best_f1:.3f}")

    # Metrics at best_t on val and test
    print("\nMetrics at best threshold")
    val_metrics = metrics_for(df_val, best_t, "VAL")
    test_metrics = metrics_for(df_test, best_t, "TEST")

    out = {
        "threshold_sweep": history,
        "best_threshold": float(best_t),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    out_path = metrics_dir / "baseline_rf_threshold_tuned.json"
    with out_path.open("w", encoding="utf8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved RF threshold tuning results to {out_path}")


if __name__ == "__main__":
    main()
