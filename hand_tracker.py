import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """Class to manage MediaPipe Hand Tracking detection and landmark processing (21 Landmarks)."""

    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # Thumb
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # Index Finger
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # Middle Finger
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # Ring Finger
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),  # Palm & Pinky Finger
    ]

    FINGERTIP_IDS = [4, 8, 12, 16, 20]

    def __init__(
        self,
        model_path="hand_landmarker.task",
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_results = None

    def process_frame(self, frame):
        """Processes the input BGR image frame and detects hand landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        self.last_results = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )
        return self.last_results

    def get_hand_points(self, frame, results=None):
        """Returns a list of 21 landmark pixel coordinates for each detected hand."""
        if results is None:
            results = self.last_results

        if not results or not results.hand_landmarks:
            return []

        h, w, _ = frame.shape
        hands_points = []

        for landmarks in results.hand_landmarks:
            points = []
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
            hands_points.append(points)

        return hands_points

    def draw_landmarks(self, frame, results=None, is_mirrored=True):
        """Draws 21 hand landmark points and skeletal connection lines on the frame.

        is_mirrored: Swaps 'Left' <-> 'Right' label to match user's mirror view perspective.
        """
        if results is None:
            results = self.last_results

        if not results or not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand_idx, landmarks in enumerate(results.hand_landmarks):
            points = []
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            # 1. Draw connection lines between landmark points
            for p1_idx, p2_idx in self.HAND_CONNECTIONS:
                pt1 = points[p1_idx]
                pt2 = points[p2_idx]
                cv2.line(frame, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

            # 2. Draw 21 joint landmark nodes
            for idx, (cx, cy) in enumerate(points):
                if idx in self.FINGERTIP_IDS:
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(
                        frame, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA
                    )
                else:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

            # 3. Mirror-correct and render Handedness Label (Right / Left)
            if results.handedness and hand_idx < len(results.handedness):
                hand_info = results.handedness[hand_idx][0]
                raw_label = hand_info.category_name
                score = int(hand_info.score * 100)

                if is_mirrored:
                    display_label = "Right" if raw_label == "Left" else "Left"
                else:
                    display_label = "Left" if raw_label == "Left" else "Right"

                wrist_x, wrist_y = points[0]
                cv2.putText(
                    frame,
                    f"{display_label} ({score}%)",
                    (wrist_x - 35, wrist_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame

    def close(self):
        """Closes the underlying MediaPipe detector resource."""
        if self.detector:
            self.detector.close()
