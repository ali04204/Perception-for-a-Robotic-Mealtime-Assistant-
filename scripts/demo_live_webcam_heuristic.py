import collections

import cv2
import mediapipe as mp
import numpy as np


# Simple smoothing over last N frames
class Smoother:
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)

    def update(self, value):
        self.buffer.append(value)
        if not self.buffer:
            return value
        ones = sum(self.buffer)
        zeros = len(self.buffer) - ones
        return 1 if ones >= zeros else 0


def main():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # make the OpenCV window resizable and start a bit larger
    window_name = "Hygia Robot live readiness demo (heuristic)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    smoother = Smoother(window_size=9)
    last_pred = 0

    # Hand to mouth distance threshold in normalized image units
    ready_dist_thresh = 0.22

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)
            rgb.flags.writeable = True

            face_present = result.face_landmarks is not None

            # Default to last prediction
            instant_pred = last_pred
            min_dist = None

            mouth_xy_norm = None
            right_tip_xy_norm = None
            left_tip_xy_norm = None
            right_dist = None
            left_dist = None

            # 1) compute hand to mouth distances if we see a face and any hand
            if face_present and (
                result.right_hand_landmarks is not None
                or result.left_hand_landmarks is not None
            ):
                # upper lip area as mouth reference
                mouth_lm = result.face_landmarks.landmark[13]
                mouth_xy_norm = np.array([mouth_lm.x, mouth_lm.y])

                if result.right_hand_landmarks is not None:
                    rh = result.right_hand_landmarks
                    rh_tip = rh.landmark[8]
                    right_tip_xy_norm = np.array([rh_tip.x, rh_tip.y])
                    right_dist = np.linalg.norm(
                        mouth_xy_norm - right_tip_xy_norm
                    )

                if result.left_hand_landmarks is not None:
                    lh = result.left_hand_landmarks
                    lh_tip = lh.landmark[8]
                    left_tip_xy_norm = np.array([lh_tip.x, lh_tip.y])
                    left_dist = np.linalg.norm(
                        mouth_xy_norm - left_tip_xy_norm
                    )

                d_candidates = [
                    d for d in [right_dist, left_dist] if d is not None
                ]
                if d_candidates:
                    min_dist = min(d_candidates)
                    instant_pred = 1 if min_dist < ready_dist_thresh else 0
                    last_pred = instant_pred

            # If we see a face but no hands at all, treat as NOT READY
            if face_present and (
                result.right_hand_landmarks is None
                and result.left_hand_landmarks is None
            ):
                instant_pred = 0
                last_pred = 0

            # If no face, also treat as NOT READY
            if not face_present:
                instant_pred = 0
                last_pred = 0


            # 2) temporal smoothing
            smooth_pred = smoother.update(instant_pred)

            # 3) draw landmarks for context
            if face_present:
                mp_drawing.draw_landmarks(
                    frame,
                    result.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1,
                        circle_radius=1,
                    ),
                )

            if result.right_hand_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame,
                    result.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )

            if result.left_hand_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame,
                    result.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )

            # 4) draw mouth point and hand lines with per hand distances
            if mouth_xy_norm is not None:
                mx = int(mouth_xy_norm[0] * w)
                my = int(mouth_xy_norm[1] * h)
                cv2.circle(frame, (mx, my), 6, (255, 255, 0), 2)  # mouth

                # Right hand
                if right_tip_xy_norm is not None:
                    rx = int(right_tip_xy_norm[0] * w)
                    ry = int(right_tip_xy_norm[1] * h)
                    cv2.circle(frame, (rx, ry), 6, (0, 255, 255), -1)
                    if right_dist is not None:
                        col_r = (
                            (0, 255, 0)
                            if right_dist < ready_dist_thresh
                            else (0, 0, 255)
                        )
                        cv2.line(frame, (mx, my), (rx, ry), col_r, 2)
                        cv2.putText(
                            frame,
                            f"R d={right_dist:.3f}",
                            (rx + 5, ry - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            col_r,
                            1,
                            cv2.LINE_AA,
                        )

                # Left hand
                if left_tip_xy_norm is not None:
                    lx = int(left_tip_xy_norm[0] * w)
                    ly = int(left_tip_xy_norm[1] * h)
                    cv2.circle(frame, (lx, ly), 6, (0, 165, 255), -1)
                    if left_dist is not None:
                        col_l = (
                            (0, 255, 0)
                            if left_dist < ready_dist_thresh
                            else (0, 0, 255)
                        )
                        cv2.line(frame, (mx, my), (lx, ly), col_l, 2)
                        cv2.putText(
                            frame,
                            f"L d={left_dist:.3f}",
                            (lx + 5, ly - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            col_l,
                            1,
                            cv2.LINE_AA,
                        )
                        # 5) final label and overlays
            label = "READY" if smooth_pred == 1 else "NOT READY"
            color = (0, 255, 0) if smooth_pred == 1 else (0, 0, 255)

            # top info bar
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

            if not face_present:
                info_text = "Face not detected"
            elif min_dist is not None:
                info_text = (
                    f"Prediction: {label}   d_min={min_dist:.3f}   "
                    f"thresh={ready_dist_thresh:.2f}"
                )
            else:
                info_text = (
                    f"Prediction: {label}   d_min=n/a   "
                    f"thresh={ready_dist_thresh:.2f}"
                )

            # main status text
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            # small legend under the top bar
            cv2.putText(
                frame,
                "Yellow = mouth   Cyan / orange = hands",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # bottom coloured bar
            bar_height = 40
            cv2.rectangle(
                frame,
                (0, h - bar_height),
                (w, h),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(
                frame,
                (0, h - bar_height),
                (w, h),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
