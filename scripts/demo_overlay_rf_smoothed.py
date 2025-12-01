import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd


def load_predictions_for_clip(
    repo_root: pathlib.Path,
    clip_id: str,
    preds_csv_name: str,
) -> np.ndarray:
    """
    Load smoothed window level predictions for a given clip
    from results/predictions/{preds_csv_name}.

    The CSV is expected to contain:
      - clip_id column
      - y_pred_smooth column (0 or 1)
    """
    preds_path = repo_root / "results" / "predictions" / preds_csv_name
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {preds_path}")

    df = pd.read_csv(preds_path)

    if "clip_id" not in df.columns:
        raise ValueError(
            f"Expected 'clip_id' column in {preds_csv_name}, got columns: {list(df.columns)}"
        )

    df_clip = df[df["clip_id"].astype(str) == clip_id].copy()
    if df_clip.empty:
        raise ValueError(f"No smoothed predictions found for clip_id {clip_id!r} in {preds_csv_name}")

    if "y_pred_smooth" not in df_clip.columns:
        raise ValueError(
            f"Expected 'y_pred_smooth' column in {preds_csv_name}, got columns: {list(df.columns)}"
        )

    # Ensure a stable order
    df_clip = df_clip.reset_index(drop=True)
    return df_clip["y_pred_smooth"].astype(int).to_numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Play a raw clip with smoothed ready/not_ready overlay."
    )
    parser.add_argument(
        "clip_id",
        help="Clip id, e.g. 39 or 52. Must match the name of data/raw/{clip_id}.mp4",
    )
    parser.add_argument(
        "--preds-csv",
        default="baseline_rf_test_windows_smoothed.csv",
        help="Filename under results/predictions/ to use for smoothed window predictions.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, also save an MP4 with the overlay to results/demo/",
    )
    args = parser.parse_args()

    clip_id = str(args.clip_id)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    video_path = repo_root / "data" / "raw" / f"{clip_id}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    preds = load_predictions_for_clip(repo_root, clip_id, args.preds_csv)
    num_windows = len(preds)

    print(f"Loaded {num_windows} windows of smoothed predictions for clip {clip_id} "
          f"from {args.preds_csv}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {frame_count} frames at {fps:.2f} fps, size {width}x{height}")

    bar_height = 40
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Optional writer
    writer = None
    if args.save:
        out_dir = repo_root / "results" / "demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_id}_overlay_{args.preds_csv.replace('.csv', '')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height + bar_height))
        print(f"Saving overlay video to {out_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Map current frame to a window index
        if frame_count > 0 and num_windows > 0:
            window_idx = int(frame_idx * num_windows / frame_count)
            if window_idx >= num_windows:
                window_idx = num_windows - 1
            pred = preds[window_idx]
        else:
            pred = 0  # default not_ready

        label_str = "READY" if pred == 1 else "NOT READY"

        # Create a bar below the frame
        bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

        if pred == 1:
            # green-ish for ready
            bar[:] = (0, 180, 0)
        else:
            # red-ish for not_ready
            bar[:] = (0, 0, 180)

        # Put text in the bar (model agnostic)
        text = f"Smoothed prediction: {label_str}"
        cv2.putText(
            bar,
            text,
            (10, int(bar_height * 0.7)),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined = np.vstack([frame, bar])

        cv2.imshow(f"Clip {clip_id} with smoothed overlay", combined)

        if writer is not None:
            writer.write(combined)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:  # ESC
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
