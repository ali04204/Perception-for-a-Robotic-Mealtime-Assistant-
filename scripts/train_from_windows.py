import json
import pathlib
import sys
from datetime import datetime

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from feature_config import WINDOW_FEATURE_COLUMNS


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_WINDOWS_CSV = ROOT / "results" / "features" / "all_windows.csv"
MODELS_DIR = ROOT / "results" / "models"
CONFIG_DIR = ROOT / "results" / "config"


def get_git_commit() -> str | None:
    """Return current git commit hash if available."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def main():
    # Choose dataset path
    if len(sys.argv) > 1:
        ds_path = pathlib.Path(sys.argv[1]).resolve()
    else:
        ds_path = DEFAULT_WINDOWS_CSV

    assert ds_path.exists(), f"Missing dataset csv: {ds_path}"
    print(f"Loading windows dataset from: {ds_path}")

    ds = pd.read_csv(ds_path)

    # Basic checks
    for col in ["split", "label"]:
        assert col in ds.columns, f"Expected column '{col}' in {ds_path}"

    # Filter to train split only
    train_df = ds[ds["split"] == "train"].copy()
    assert not train_df.empty, "No rows with split == 'train' in dataset"

    # Labels  not_ready -> 0, ready -> 1
    label_map = {"not_ready": 0, "ready": 1}
    unknown_labels = set(train_df["label"].unique()) - set(label_map.keys())
    if unknown_labels:
        raise ValueError(f"Unexpected labels in data: {unknown_labels}")

    y_train = train_df["label"].map(label_map).astype(int)

    # Features from frozen list
    missing = [c for c in WINDOW_FEATURE_COLUMNS if c not in train_df.columns]
    if missing:
        print("Error: dataset is missing expected feature columns:")
        for c in missing:
            print("  ", c)
        raise SystemExit(1)

    X_train = train_df[WINDOW_FEATURE_COLUMNS].values

    print(f"Train rows: {X_train.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")

    # Define RF
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )

    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    print("Training complete.")

    # Ensure output dirs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "baseline_rf.joblib"
    dump(rf, model_path)
    print(f"Saved RF model to: {model_path}")

    # Save config JSON
    config = {
        "dataset_path": str(ds_path.relative_to(ROOT)),
        "features": list(WINDOW_FEATURE_COLUMNS),
        "label_map": label_map,
        "rf_params": rf.get_params(),
        "created_at": datetime.now().isoformat(),
    }

    commit = get_git_commit()
    if commit is not None:
        config["git_commit"] = commit

    config_path = CONFIG_DIR / "baseline_rf.json"
    with config_path.open("w", encoding="utf8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved RF config to: {config_path}")


if __name__ == "__main__":
    main()
