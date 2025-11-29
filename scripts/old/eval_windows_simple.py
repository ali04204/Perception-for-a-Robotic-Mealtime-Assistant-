import sys
import pathlib

import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_windows_simple.py PATH_TO_WINDOWS_CSV")
        sys.exit(1)

    ds_path = pathlib.Path(sys.argv[1]).resolve()
    assert ds_path.exists(), f"Missing dataset csv: {ds_path}"

    ds = pd.read_csv(ds_path)

    # True labels as strings
    y_true = ds["label"].astype(str)

    # Features  must match what train_from_windows.py used
    X = ds.drop(columns=["label", "t0", "t1"])

    # Load the matching model saved by train_from_windows.py
    model_path = ds_path.parent / f"{ds_path.stem}_rf.joblib"
    assert model_path.exists(), f"Missing model file: {model_path}"
    clf = load(model_path)

    y_pred_int = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]

    id2label = {0: "not_ready", 1: "ready"}
    y_pred = [id2label[int(v)] for v in y_pred_int]

    t0_series = ds["t0"]
    t1_series = ds["t1"]

    print(f"Using dataset: {ds_path.name}")
    print(f"Using model:   {model_path.name}")
    print()
    print("t0_sec  t1_sec   true        pred        p_ready")
    print("------------------------------------------------")

    for t0, t1, yt, yp, pr in zip(t0_series, t1_series, y_true, y_pred, proba):
        print(f"{t0:6.2f}  {t1:6.2f}   {yt:10s}  {yp:10s}  {pr:7.3f}")

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["not_ready", "ready"])
    print()
    print(f"Window level accuracy on this file: {acc:.3f}")
    print("Confusion matrix [rows=true, cols=pred]:")
    print("          not_ready  ready")
    print(f"not_ready   {cm[0,0]:9d}  {cm[0,1]:5d}")
    print(f"ready       {cm[1,0]:9d}  {cm[1,1]:5d}")


if __name__ == "__main__":
    main()
