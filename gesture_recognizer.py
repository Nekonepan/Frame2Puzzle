import math
import cv2
import numpy as np


class GestureRecognizer:
    """Kelas untuk mengenali gestur tangan berbasis koordinat 21 Landmark MediaPipe"""

    # Nama-nama gestur yang didukung
    GESTURE_NONE = "NONE"
    GESTURE_PINCH = "PINCH"
    GESTURE_OPEN_PALM = "OPEN_PALM"
    GESTURE_FRAME = "FRAME"

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
        """Mendeteksi gestur pada satu tangan (PINCH, OPEN_PALM, atau NONE)"""
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

        # MENCEGAH FALSE POSITIVE SAAT MENGEPAL:
        # Saat tangan mengepal, ujung telunjuk terlipat ke dalam telapak dekat wrist.
        # Pada gestur Pinch asli, jari telunjuk terentang menjauhi pergelangan (#0).
        dist_index_wrist = self._euclidean_distance(index_tip, wrist)
        dist_mcp_wrist = self._euclidean_distance(index_mcp, wrist)
        is_index_extended = dist_index_wrist > dist_mcp_wrist * 1.05

        # Pinch valid jika jaraknya dekat DAN jari telunjuk dalam keadaan terentang (bukan mengepal)
        is_pinch = (pinch_dist <= self.pinch_threshold_px) and is_index_extended

        # 2. Cek status terentang jari-jari
        index_open = self.check_finger_extended(points, 8, 6, 5)
        middle_open = self.check_finger_extended(points, 12, 10, 9)
        ring_open = self.check_finger_extended(points, 16, 14, 13)
        pinky_open = self.check_finger_extended(points, 20, 18, 17)

        dist_thumb_pinky = self._euclidean_distance(thumb_tip, pinky_mcp)
        thumb_open = dist_thumb_pinky > self._euclidean_distance(points[2], pinky_mcp) * 0.8

        open_fingers_count = sum([index_open, middle_open, ring_open, pinky_open])

        extra_info = {
            "pinch_dist": pinch_dist,
            "pinch_center": (
                (thumb_tip[0] + index_tip[0]) // 2,
                (thumb_tip[1] + index_tip[1]) // 2,
            ),
            "open_count": open_fingers_count,
            "is_pinch": is_pinch,
            "index_open": index_open,
            "thumb_open": thumb_open,
            "middle_open": middle_open,
            "ring_open": ring_open,
            "pinky_open": pinky_open,
        }

        # Keputusan Gestur Tunggal (Prioritas: PINCH -> OPEN_PALM -> NONE)
        if is_pinch:
            return self.GESTURE_PINCH, extra_info

        if open_fingers_count >= 4 and thumb_open:
            return self.GESTURE_OPEN_PALM, extra_info

        return self.GESTURE_NONE, extra_info

    def is_l_shape(self, points, info):
        """Mengecek apakah satu tangan membentuk pola L (Ibu jari + Telunjuk terbuka)"""
        index_open = info.get("index_open", False)

        dist_thumb_wrist = self._euclidean_distance(points[4], points[0])
        dist_mcp_wrist = self._euclidean_distance(points[2], points[0])
        thumb_open = dist_thumb_wrist > dist_mcp_wrist * 1.1

        ring_open = info.get("ring_open", False)
        pinky_open = info.get("pinky_open", False)
        others_mostly_folded = not (ring_open and pinky_open)

        return index_open and thumb_open and others_mostly_folded

    def detect_frame_gesture(self, hands_points):
        """Mendeteksi Gestur Bingkai Foto (Frame Gesture) dari 2 Tangan"""
        if len(hands_points) < 2:
            return False, None

        hand1_pts = hands_points[0]
        hand2_pts = hands_points[1]

        _, info1 = self.detect_single_hand_gesture(hand1_pts)
        _, info2 = self.detect_single_hand_gesture(hand2_pts)

        l1 = self.is_l_shape(hand1_pts, info1)
        l2 = self.is_l_shape(hand2_pts, info2)

        if l1 and l2:
            t1, i1 = hand1_pts[4], hand1_pts[8]
            t2, i2 = hand2_pts[4], hand2_pts[8]

            all_tips = [t1, i1, t2, i2]
            xs = [p[0] for p in all_tips]
            ys = [p[1] for p in all_tips]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y

            if width > 50 and height > 50:
                frame_roi = (min_x, min_y, max_x, max_y)
                return True, frame_roi

        return False, None
