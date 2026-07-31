import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """Kelas untuk mengelola deteksi dan pengolahan Hand Tracking MediaPipe (21 Landmarks)"""

    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # Ibu Jari (Thumb)
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # Jari Telunjuk (Index)
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # Jari Tengah (Middle)
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # Jari Manis (Ring)
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),  # Telapak & Jari Kelingking (Pinky)
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
        """Memproses frame gambar (BGR) dan mendeteksi landmark tangan"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        self.last_results = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )
        return self.last_results

    def draw_landmarks(self, frame, results=None, is_mirrored=True):
        """Menggambar 21 titik landmark tangan dan garis koneksi pada frame

        is_mirrored: Jika True (default), balikkan label 'Left' <-> 'Right'
        agar sesuai dengan tampilan cermin pengguna.
        """
        if results is None:
            results = self.last_results

        if not results or not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand_idx, landmarks in enumerate(results.hand_landmarks):
            # 1. Konversi koordinat normalisasi ke piksel layar
            points = []
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            # 2. Gambar garis koneksi antar titik tangan
            for p1_idx, p2_idx in self.HAND_CONNECTIONS:
                pt1 = points[p1_idx]
                pt2 = points[p2_idx]
                cv2.line(frame, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)

            # 3. Gambar 21 titik sendi (Landmarks)
            for idx, (cx, cy) in enumerate(points):
                if idx in self.FINGERTIP_IDS:
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(
                        frame, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA
                    )
                else:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

            # 4. Koreksi dan Tampilkan Label Handedness (Kanan / Kiri)
            if results.handedness and hand_idx < len(results.handedness):
                hand_info = results.handedness[hand_idx][0]
                raw_label = hand_info.category_name  # 'Left' atau 'Right' dari sensor
                score = int(hand_info.score * 100)

                # Jika tampilan kamera di-mirror (cermin), balikkan labelnya:
                # 'Left' di sensor -> Tangan Kanan fisik pengguna di cermin
                # 'Right' di sensor -> Tangan Kiri fisik pengguna di cermin
                if is_mirrored:
                    display_label = (
                        "Kanan" if raw_label == "Left" else "Kiri"
                    )
                else:
                    display_label = (
                        "Kiri" if raw_label == "Left" else "Kanan"
                    )

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
        """Menutup detector resource"""
        if self.detector:
            self.detector.close()
