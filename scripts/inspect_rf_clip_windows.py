import argparse
import pathlib

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Inspect RF smoothed predictions and labels for a single clip."
    )
    parser.add_argument("clip_id", help="Clip id, for example 39.")
    parser.add_argument(
        "--preds-csv",
        default="baseline_rf_test_windows_smoothed.csv",
        help="Predictions CSV under results/predictions/.",
    )
    args = parser.parse_args()

    clip_id = str(args.clip_id)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    preds_path = repo_root / "results" / "predictions" / args.preds_csv

    if not preds_path.exists():
        raise SystemExit(f"Missing predictions CSV: {preds_path}")

    df = pd.read_csv(preds_path)

    required_cols = ["clip_id", "y_pred_smooth"]
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"Expected column {c!r} in {args.preds_csv}")

    if "label" not in df.columns:
        print("Warning: no 'label' column in predictions CSV. You will only see predictions.")
    if "p_ready_raw" not in df.columns:
        print("Warning: no 'p_ready_raw' column in predictions CSV.")

    df_clip = df[df["clip_id"].astype(str) == clip_id].copy()
    if df_clip.empty:
        raise SystemExit(
            f"No rows for clip_id {clip_id!r} in {args.preds_csv}. "
            "Check that this clip is in the test split."
        )

    df_clip = df_clip.reset_index(drop=True)

    # Simple counts
    if "label" in df_clip.columns:
        print("Label counts for clip", clip_id, ":", df_clip["label"].value_counts().to_dict())
    print("Predicted (smoothed) counts for clip", clip_id, ":", df_clip["y_pred_smooth"].value_counts().to_dict())
    print()

    has_t0 = "t0_sec" in df_clip.columns
    has_t1 = "t1_sec" in df_clip.columns

    print(f"First {len(df_clip)} windows for clip {clip_id}:")
    for i, row in df_clip.iterrows():
        true_label = row["label"] if "label" in df_clip.columns else "?"
        pred = int(row["y_pred_smooth"])
        pred_label = "READY" if pred == 1 else "NOT READY"

        line_parts = [f"[{i:02d}]"]

        if has_t0 and has_t1:
            line_parts.append(f"{row['t0_sec']:.2f}-{row['t1_sec']:.2f}s")
        elif has_t0:
            line_parts.append(f"{row['t0_sec']:.2f}s")

        line_parts.append(f"true={true_label}")
        line_parts.append(f"pred={pred_label}")

        if "p_ready_raw" in df_clip.columns:
            line_parts.append(f"p_ready_raw={row['p_ready_raw']:.3f}")

        print("  " + "  |  ".join(line_parts))


if __name__ == "__main__":
    main()
