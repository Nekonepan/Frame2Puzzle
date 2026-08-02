import math
import cv2
import numpy as np


class GestureRecognizer:
    """Kelas untuk mengenali gestur tangan berbasis koordinat 21 Landmark MediaPipe"""

    # Nama-nama gestur yang didukung
    GESTURE_NONE = "NONE"
    GESTURE_PINCH = "PINCH"
    GESTURE_OPEN_PALM = "OPEN_PALM"
    GESTURE_TWO_FINGERS = "TWO_FINGERS"  # Gestur 2 Jari (Telunjuk + Tengah) untuk Capture

    def __init__(self, pinch_threshold_px=40):
        self.pinch_threshold_px = pinch_threshold_px

    @staticmethod
    def _euclidean_distance(pt1, pt2):
        """Menghitung jarak Euclidean 2D antara dua titik (x, y)"""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def check_finger_extended(self, points, tip_idx, pip_idx, mcp_idx, wrist_idx=0):
        """Mengecek apakah suatu jari sedang terbuka/terentang lurus"""
        dist_tip_wrist = self._euclidean_distance(points[tip_idx], points[wrist_idx])
        dist_pip_wrist = self._euclidean_distance(points[pip_idx], points[wrist_idx])
        return dist_tip_wrist > dist_pip_wrist * 1.1

    def detect_single_hand_gesture(self, points):
        """Mendeteksi gestur pada satu tangan (PINCH, TWO_FINGERS, OPEN_PALM, atau NONE)"""
        if len(points) < 21:
            return self.GESTURE_NONE, {}

        wrist = points[0]
        thumb_tip, thumb_ip = points[4], points[3]
        index_tip, index_pip, index_mcp = points[8], points[6], points[5]
        middle_tip, middle_pip, middle_mcp = points[12], points[10], points[9]
        ring_tip, ring_pip, ring_mcp = points[16], points[14], points[13]
        pinky_tip, pinky_pip, pinky_mcp = points[20], points[18], points[17]

        # 1. Cek Jarak Pinch (Ujung Ibu Jari #4 ke Ujung Telunjuk #8)
        pinch_dist = self._euclidean_distance(thumb_tip, index_tip)

        dist_index_wrist = self._euclidean_distance(index_tip, wrist)
        dist_mcp_wrist = self._euclidean_distance(index_mcp, wrist)
        is_index_extended = dist_index_wrist > dist_mcp_wrist * 1.05

        is_pinch = (pinch_dist <= self.pinch_threshold_px) and is_index_extended

        # 2. Cek status terentang jari-jari
        index_open = self.check_finger_extended(points, 8, 6, 5)
        middle_open = self.check_finger_extended(points, 12, 10, 9)
        ring_open = self.check_finger_extended(points, 16, 14, 13)
        pinky_open = self.check_finger_extended(points, 20, 18, 17)

        dist_thumb_pinky = self._euclidean_distance(thumb_tip, pinky_mcp)
        thumb_open = dist_thumb_pinky > self._euclidean_distance(points[2], pinky_mcp) * 0.8

        open_fingers_count = sum([index_open, middle_open, ring_open, pinky_open])

        # Cek Gestur 2 Jari (Hanya Telunjuk & Tengah Terbuka, Manis & Kelingking Terlipat)
        is_two_fingers = index_open and middle_open and not ring_open and not pinky_open

        extra_info = {
            "pinch_dist": pinch_dist,
            "pinch_center": (
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2,
            ),
            "open_count": open_fingers_count,
            "is_pinch": is_pinch,
            "is_two_fingers": is_two_fingers,
            "index_open": index_open,
            "middle_open": middle_open,
            "thumb_open": thumb_open,
            "ring_open": ring_open,
            "pinky_open": pinky_open,
        }

        # Priority: PINCH -> TWO_FINGERS (CAPTURE) -> OPEN_PALM -> NONE
        if is_pinch:
            return self.GESTURE_PINCH, extra_info

        if is_two_fingers:
            return self.GESTURE_TWO_FINGERS, extra_info

        if open_fingers_count >= 4 and thumb_open:
            return self.GESTURE_OPEN_PALM, extra_info

        return self.GESTURE_NONE, extra_info
