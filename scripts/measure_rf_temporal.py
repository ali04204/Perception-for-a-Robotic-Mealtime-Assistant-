import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOOTHED_CSV = ROOT / "results" / "features" / "all_windows_rf_smoothed.csv"
OUT_JSON = ROOT / "results" / "metrics" / "baseline_rf_temporal.json"


def choose_time_column(df: pd.DataFrame) -> str | None:
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
    preferred = ["clip_id", "video_id", "id"]
    for col in preferred:
        if col in df.columns:
            return col
    return None


def estimate_stride_sec(df: pd.DataFrame, clip_col: str, time_col: str) -> float:
    """Estimate window stride from time column if present."""
    diffs = []
    for _, g in df.groupby(clip_col):
        t = g[time_col].values
        d = np.diff(t)
        d = d[d > 0]
        if d.size:
            diffs.append(np.median(d))
    if not diffs:
        return 1.0
    return float(np.median(diffs))


def main():
    assert SMOOTHED_CSV.exists(), f"Missing {SMOOTHED_CSV}"
    df = pd.read_csv(SMOOTHED_CSV)

    for col in ["split", "label", "rf_smoothed_label"]:
        assert col in df.columns, f"Expected column '{col}' in {SMOOTHED_CSV}"

    clip_col = choose_clip_column(df)
    assert clip_col is not None, (
        "Could not find a clip id column. "
        "Expected one of clip_id, video_id, id in smoothed CSV"
    )

    time_col = choose_time_column(df)
    if time_col is None:
        print("Warning: no explicit time column, using window index for timing.")
        df["_order_idx"] = np.arange(len(df))
        time_col = "_order_idx"

    # Only test split
    test_df = df[df["split"] == "test"].copy()
    assert not test_df.empty, "No rows with split == 'test' in smoothed CSV"

    # Estimate stride_sec
    window_stride_sec = estimate_stride_sec(test_df, clip_col, time_col)
    print(f"Estimated window_stride_sec: {window_stride_sec:.4f} s")

    per_clip_results = []
    all_latencies_windows = []

    for clip_id, g in (
        test_df.sort_values([clip_col, time_col])
        .groupby(clip_col, sort=False)
    ):
        labels_true = g["label"].map({"not_ready": 0, "ready": 1}).values
        labels_pred = g["rf_smoothed_label"].map({"not_ready": 0, "ready": 1}).values

        # Find true onsets: not_ready -> ready
        onset_idxs = []
        for i in range(1, len(labels_true)):
            if labels_true[i - 1] == 0 and labels_true[i] == 1:
                onset_idxs.append(i)

        latencies_windows = []

        for t_true in onset_idxs:
            # First index >= t_true where smoothed becomes ready
            found = np.where(labels_pred[t_true:] == 1)[0]
            if found.size == 0:
                continue
            t_pred = t_true + int(found[0])
            delta = t_pred - t_true
            if delta >= 0:
                latencies_windows.append(delta)
                all_latencies_windows.append(delta)

        # Flicker: number of label flips in smoothed sequence
        flips = 0
        for i in range(1, len(labels_pred)):
            if labels_pred[i] != labels_pred[i - 1]:
                flips += 1

        # Duration in seconds based on time_col
        t_vals = g[time_col].values
        if len(t_vals) > 1:
            duration_sec = (t_vals[-1] - t_vals[0]) + window_stride_sec
        else:
            duration_sec = window_stride_sec

        duration_min = duration_sec / 60.0
        flicker_per_min = flips / duration_min if duration_min > 0 else 0.0

        per_clip_results.append(
            {
                "clip_id": clip_id,
                "num_windows": int(len(labels_true)),
                "num_true_onsets": int(len(onset_idxs)),
                "latencies_windows": latencies_windows,
                "flicker_flips": int(flips),
                "duration_sec": float(duration_sec),
                "flicker_per_min": float(flicker_per_min),
            }
        )

    # Aggregate latency stats
    if all_latencies_windows:
        latencies_windows_arr = np.array(all_latencies_windows, dtype=float)
        mean_latency_windows = float(latencies_windows_arr.mean())
        std_latency_windows = float(latencies_windows_arr.std())
        mean_latency_sec = mean_latency_windows * window_stride_sec
        std_latency_sec = std_latency_windows * window_stride_sec
    else:
        mean_latency_windows = std_latency_windows = mean_latency_sec = std_latency_sec = None

    # Aggregate flicker
    flickers = [c["flicker_per_min"] for c in per_clip_results]
    if flickers:
        mean_flicker_per_min = float(np.mean(flickers))
        std_flicker_per_min = float(np.std(flickers))
    else:
        mean_flicker_per_min = std_flicker_per_min = None

    out = {
        "window_stride_sec_estimate": window_stride_sec,
        "per_clip": per_clip_results,
        "aggregate": {
            "mean_latency_windows": mean_latency_windows,
            "std_latency_windows": std_latency_windows,
            "mean_latency_sec": mean_latency_sec,
            "std_latency_sec": std_latency_sec,
            "mean_flicker_per_min": mean_flicker_per_min,
            "std_flicker_per_min": std_flicker_per_min,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved RF temporal metrics to: {OUT_JSON}")
    print("Aggregate:")
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
