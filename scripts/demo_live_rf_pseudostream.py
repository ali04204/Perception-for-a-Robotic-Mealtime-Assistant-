import argparse
import pathlib
import time

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Pseudo live console demo using RF smoothed window predictions."
    )
    parser.add_argument(
        "clip_id",
        help="Clip id, for example 39. Must exist in the predictions CSV.",
    )
    parser.add_argument(
        "--preds-csv",
        default="baseline_rf_test_windows_smoothed.csv",
        help="Predictions CSV under results/predictions/. Default is RF smoothed.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to pause between windows to simulate streaming.",
    )
    args = parser.parse_args()

    clip_id = str(args.clip_id)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    preds_path = repo_root / "results" / "predictions" / args.preds_csv

    if not preds_path.exists():
        raise SystemExit(f"Missing predictions CSV: {preds_path}")

    df = pd.read_csv(preds_path)

    if "clip_id" not in df.columns:
        raise SystemExit(f"Expected 'clip_id' column in {args.preds_csv}")
    if "y_pred_smooth" not in df.columns:
        raise SystemExit(f"Expected 'y_pred_smooth' column in {args.preds_csv}")

    has_t0 = "t0_sec" in df.columns
    has_t1 = "t1_sec" in df.columns

    df_clip = df[df["clip_id"].astype(str) == clip_id].copy()
    if df_clip.empty:
        raise SystemExit(
            f"No rows for clip_id {clip_id!r} in {args.preds_csv}. "
            "Check that this clip is in the test split."
        )

    df_clip = df_clip.reset_index(drop=True)

    print(f"Pseudo live RF demo for clip {clip_id}")
    print(f"Source CSV: {preds_path}")
    print(f"Total windows: {len(df_clip)}")
    print("Streaming predictions...\n")

    for i, row in df_clip.iterrows():
        pred = int(row["y_pred_smooth"])
        label = "READY" if pred == 1 else "NOT READY"

        if has_t0 and has_t1:
            print(
                f"[{i:02d}] {row['t0_sec']:.2f} - {row['t1_sec']:.2f} s  ->  {label}"
            )
        elif has_t0:
            print(
                f"[{i:02d}] {row['t0_sec']:.2f} s  ->  {label}"
            )
        else:
            print(
                f"[{i:02d}] window {i}  ->  {label}"
            )

        time.sleep(args.sleep)

    print("\nDone.")


if __name__ == "__main__":
    main()
