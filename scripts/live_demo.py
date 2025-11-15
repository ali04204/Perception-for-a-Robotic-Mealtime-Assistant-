import sys
import pathlib
import cv2
import numpy as np
import joblib
import mediapipe as mp


FEATURE_NAMES = [
    "lh_to_mouth",
    "rh_to_mouth",
    "lh_speed",
    "rh_speed",
    "lwrist_rel_shoulder_y",
    "rwrist_rel_shoulder_y",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def angle_deg(a, b, c):
    if any(np.isnan(v) for v in [*a, *b, *c]):
        return np.nan
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    feat_dir = repo / "results" / "features"
    model_path = feat_dir / "baseline_rf.joblib"
    assert model_path.exists(), f"Missing model {model_path}"

    clf = joblib.load(model_path)

    # video source
    if len(sys.argv) > 1:
        src = sys.argv[1]
        cap = cv2.VideoCapture(src)
        print(f"Using video file {src}")
    else:
        cap = cv2.VideoCapture(0)
        print("Using webcam 0")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    mp_holistic = mp.solutions.holistic

    prev_lh = None
    prev_rh = None

    with mp_holistic.Holistic(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            # draw pose and hands on the BGR frame
            frame.flags.writeable = True
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    res.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                )

            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    res.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                )

            pred_label = "no_pose"
            p_ready = 0.0

            if res.pose_landmarks:
                pose = res.pose_landmarks.landmark

                # pose points
                def pxy(idx):
                    lm = pose[idx]
                    return lm.x, lm.y

                p9x, p9y = pxy(9)
                p10x, p10y = pxy(10)
                mouth_x = (p9x + p10x) / 2.0
                mouth_y = (p9y + p10y) / 2.0

                lwx, lwy = pxy(15)
                rwx, rwy = pxy(16)

                lsx, lsy = pxy(11)
                rsx, rsy = pxy(12)
                shoulder_y = (lsy + rsy) / 2.0

                lex_x, lex_y = pxy(13)
                rex_x, rex_y = pxy(14)

                # hand landmarks may be missing
                lh_present = res.left_hand_landmarks is not None
                rh_present = res.right_hand_landmarks is not None

                if lh_present:
                    lh_lm = res.left_hand_landmarks.landmark[8]
                    lhx = lh_lm.x
                    lhy = lh_lm.y
                else:
                    lhx = np.nan
                    lhy = np.nan

                if rh_present:
                    rh_lm = res.right_hand_landmarks.landmark[8]
                    rhx = rh_lm.x
                    rhy = rh_lm.y
                else:
                    rhx = np.nan
                    rhy = np.nan

                # distances to mouth
                lh_to_mouth = np.sqrt((lhx - mouth_x) ** 2 + (lhy - mouth_y) ** 2)
                rh_to_mouth = np.sqrt((rhx - mouth_x) ** 2 + (rhy - mouth_y) ** 2)

                # wrist height relative to shoulders
                lwrist_rel_shoulder_y = lwy - shoulder_y
                rwrist_rel_shoulder_y = rwy - shoulder_y

                # elbow angles
                left_elbow_angle_deg = angle_deg(
                    (lsx, lsy), (lex_x, lex_y), (lwx, lwy)
                )
                right_elbow_angle_deg = angle_deg(
                    (rsx, rsy), (rex_x, rex_y), (rwx, rwy)
                )

                # speeds
                if prev_lh is None or np.any(np.isnan([lhx, lhy])):
                    lh_speed = 0.0
                else:
                    lh_speed = (
                        np.sqrt((lhx - prev_lh[0]) ** 2 + (lhy - prev_lh[1]) ** 2)
                        * fps
                    )

                if prev_rh is None or np.any(np.isnan([rhx, rhy])):
                    rh_speed = 0.0
                else:
                    rh_speed = (
                        np.sqrt((rhx - prev_rh[0]) ** 2 + (rhy - prev_rh[1]) ** 2)
                        * fps
                    )

                prev_lh = (lhx, lhy)
                prev_rh = (rhx, rhy)

                feat_vec = np.array(
                    [
                        lh_to_mouth,
                        rh_to_mouth,
                        lh_speed,
                        rh_speed,
                        lwrist_rel_shoulder_y,
                        rwrist_rel_shoulder_y,
                        left_elbow_angle_deg,
                        right_elbow_angle_deg,
                    ],
                    dtype=float,
                )

                if np.any(np.isnan(feat_vec)):
                    pred_label = "unknown"
                    p_ready = 0.0
                else:
                    X = feat_vec.reshape(1, -1)
                    pred = clf.predict(X)[0]
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X)[0]
                        p_ready = float(proba[1])
                    else:
                        p_ready = 0.0

                    pred_label = "READY" if pred == 1 else "NOT_READY"

            # overlay text
            text = f"{pred_label}  p_ready={p_ready:.2f}"
            color = (0, 255, 0) if pred_label == "READY" else (0, 0, 255)
            cv2.putText(
                frame,
                text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Live readiness demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
