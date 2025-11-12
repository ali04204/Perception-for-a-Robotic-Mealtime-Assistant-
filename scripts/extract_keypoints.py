import sys
import cv2
import numpy as np
import pandas as pd
import pathlib
import time
import yaml

def build_columns():
    cols = ["frame", "time_sec"]
    # Pose 33 landmarks with x y z visibility
    for i in range(33):
        cols += [f"p{i}_x", f"p{i}_y", f"p{i}_z", f"p{i}_v"]
    # Left hand 21 landmarks with x y z
    for i in range(21):
        cols += [f"lh{i}_x", f"lh{i}_y", f"lh{i}_z"]
    # Right hand 21 landmarks with x y z
    for i in range(21):
        cols += [f"rh{i}_x", f"rh{i}_y", f"rh{i}_z"]
    return cols

def main():
    # Resolve paths
    repo = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = repo / "configs" / "default.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    results_dir = repo / cfg["paths"]["results"] / "features"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Input video
    in_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "data" / "raw" / "demo.mp4"
    assert in_path.exists(), f"Input not found: {in_path}"

    # Lazy import mediapipe to avoid slow startup if missing
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["dataset"]["fps"]
    out_rows = []
    cols = build_columns()

    start_time = time.time()
    with mp_holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            row = [frame_idx, frame_idx / fps]

            # Pose 33
            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
            else:
                row += [np.nan] * (33 * 4)

            # Left hand 21
            if res.left_hand_landmarks:
                for lm in res.left_hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
            else:
                row += [np.nan] * (21 * 3)

            # Right hand 21
            if res.right_hand_landmarks:
                for lm in res.right_hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
            else:
                row += [np.nan] * (21 * 3)

            out_rows.append(row)
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames")

    cap.release()

    df = pd.DataFrame(out_rows, columns=cols)
    out_csv = results_dir / f"{in_path.stem}_holistic.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
