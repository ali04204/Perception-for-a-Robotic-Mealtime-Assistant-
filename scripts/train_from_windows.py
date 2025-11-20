import sys
import pathlib

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_from_windows.py PATH_TO_WINDOWS_CSV")
        sys.exit(1)

    ds_path = pathlib.Path(sys.argv[1]).resolve()
    assert ds_path.exists(), f"Missing dataset csv: {ds_path}"

    ds = pd.read_csv(ds_path)

    # Labels  map to integers 0 and 1
    y = ds["label"].map({"not_ready": 0, "ready": 1}).astype(int)

    # Features  everything except label and window times
    X = ds.drop(columns=["label", "t0", "t1"])

    print(f"Training on {ds_path.name}")
    print(f"Feature columns ({len(X.columns)}): {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=0,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["not_ready", "ready"]))

    # Save model next to the dataset with a name tied to this csv
    model_path = ds_path.parent / f"{ds_path.stem}_rf.joblib"
    dump(clf, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
