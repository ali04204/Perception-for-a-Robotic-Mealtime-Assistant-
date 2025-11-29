import json
import pathlib

import numpy as np
import pandas as pd
from joblib import load

from feature_config import WINDOW_FEATURE_COLUMNS


ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOWS_CSV = ROOT / "results" / "features" / "all_windows.csv"
MODEL_PATH = ROOT / "results" / "models" / "baseline_rf.joblib"
OUT_CSV = ROOT / "results" / "features" / "all_windows_rf_smoothed.csv"
CONFIG_PATH = ROOT / "results" / "config" / "smoothing_rf.json"


def choose_time_column(df: pd.DataFrame) -> str | None:
    """Pick a column that defines window order in time."""
    preferred = [
        "window_start_sec",
        "window_start_frame",
        "start_frame",
        "t_start",
        "frame_idx",
        "frame_index",
        "window_index",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def choose_clip_column(df: pd.DataFrame) -> str | None:
    """Pick a column that identifies clips or videos."""
    preferred = ["clip_id", "video_id", "id"]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def smooth_sequence(
    probs: np.ndarray,
    enter_threshold: float,
    exit_threshold: float,
    min_windows_enter: int,
    min_windows_exit: int,
) -> list[str]:
    """Apply hysteresis smoothing to a probability sequence."""
    state = "not_ready"
    above = 0
    below = 0
    labels: list[str] = []

    for p in probs:
        if p >= enter_threshold:
            above += 1
            below = 0
        elif p <= exit_threshold:
            below += 1
            above = 0
        else:
            above = 0
            below = 0

        if state == "not_ready" and above >= min_windows_enter:
            state = "ready"
        elif state == "ready" and below >= min_windows_exit:
            state = "not_ready"

        labels.append(state)

    return labels


def main():
    assert WINDOWS_CSV.exists(), f"Missing {WINDOWS_CSV}"
    assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"

    print(f"Loading windows dataset from: {WINDOWS_CSV}")
    df = pd.read_csv(WINDOWS_CSV)

    for col in ["split", "label"]:
        assert col in df.columns, f"Expected column {col} in all_windows.csv"

    clip_col = choose_clip_column(df)
    assert clip_col is not None, (
        "Could not find a clip id column. "
        "Expected one of clip_id, video_id, id in all_windows.csv"
    )

    time_col = choose_time_column(df)
    if time_col is None:
        print("Warning: no explicit time column found, using CSV row order")
        df["_order_idx"] = np.arange(len(df))
        time_col = "_order_idx"
    else:
        print(f"Using time column: {time_col}")

    print(f"Loading RF model from: {MODEL_PATH}")
    rf = load(MODEL_PATH)

    missing = [c for c in WINDOW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing}")

    X_all = df[WINDOW_FEATURE_COLUMNS].values
    proba_all = rf.predict_proba(X_all)[:, 1]
    df["rf_p_ready"] = proba_all

    # Smoothing parameters
    enter_threshold = 0.8
    exit_threshold = 0.2
    min_windows_enter = 2
    min_windows_exit = 2

    smoothed = np.empty(len(df), dtype=object)

    # Sort by clip and time for consistent sequences
    df_sorted = df.sort_values([clip_col, time_col]).reset_index()
    idx_col = "index"

    print("Applying smoothing per clip...")
    for clip_id, g in df_sorted.groupby(clip_col):
        idxs = g[idx_col].to_numpy()
        probs = g["rf_p_ready"].to_numpy()
        labels = smooth_sequence(
            probs,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            min_windows_enter=min_windows_enter,
            min_windows_exit=min_windows_exit,
        )
        smoothed[idxs] = labels

    df["rf_smoothed_label"] = smoothed

    # Save combined CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved RF smoothed windows to: {OUT_CSV}")

    # Save config
    config = {
        "enter_threshold": enter_threshold,
        "exit_threshold": exit_threshold,
        "min_windows_enter": min_windows_enter,
        "min_windows_exit": min_windows_exit,
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved smoothing config to: {CONFIG_PATH}")

    # Small summary
    counts = df["rf_smoothed_label"].value_counts()
    print("\nSmoothed label distribution:")
    print(counts)


if __name__ == "__main__":
    main()
