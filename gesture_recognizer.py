import math
from collections import deque
import cv2
import numpy as np


class GestureRecognizer:
    """Class for recognizing hand gestures based on MediaPipe 21 Landmark coordinates with Temporal Smoothing."""

    GESTURE_NONE = "NONE"
    GESTURE_GRAB = "GRAB"
    GESTURE_OPEN_PALM = "OPEN_PALM"
    GESTURE_TWO_FINGERS = "TWO_FINGERS"

    def __init__(self, grab_spread_threshold_px=55, history_len=3):
        # Maximum allowed spread (distance from any fingertip to centroid) for GRAB detection
        self.grab_spread_threshold_px = grab_spread_threshold_px
        # Temporal gesture history buffer for debouncing 1-frame noise
        self.gesture_history = deque(maxlen=history_len)

    @staticmethod
    def _euclidean_distance(pt1, pt2):
        """Calculates 2D Euclidean distance between two points (x, y)."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def check_finger_extended(self, points, tip_idx, pip_idx, mcp_idx, wrist_idx=0):
        """Checks if a specific finger is extended straight away from the wrist."""
        dist_tip_wrist = self._euclidean_distance(points[tip_idx], points[wrist_idx])
        dist_pip_wrist = self._euclidean_distance(points[pip_idx], points[wrist_idx])
        return dist_tip_wrist > dist_pip_wrist * 1.12

    def detect_single_hand_raw(self, points):
        """Pure geometric evaluation of single hand landmarks without temporal smoothing."""
        if len(points) < 21:
            return self.GESTURE_NONE, {}

        wrist = points[0]
        thumb_tip, thumb_ip, thumb_mcp = points[4], points[3], points[2]
        index_tip, index_pip, index_mcp = points[8], points[6], points[5]
        middle_tip, middle_pip, middle_mcp = points[12], points[10], points[9]
        ring_tip, ring_pip, ring_mcp = points[16], points[14], points[13]
        pinky_tip, pinky_pip, pinky_mcp = points[20], points[18], points[17]

        # 1. GRAB Detection: All 5 fingertips must be clustered tightly together
        fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        centroid_x = sum(pt[0] for pt in fingertips) // 5
        centroid_y = sum(pt[1] for pt in fingertips) // 5
        grab_center = (centroid_x, centroid_y)

        # Calculate max distance from any fingertip to the centroid (spread radius)
        grab_spread = max(self._euclidean_distance(tip, grab_center) for tip in fingertips)

        # GRAB is TRUE only when ALL 5 fingertips are within the spread threshold
        is_grab = grab_spread <= self.grab_spread_threshold_px

        # 2. Check finger extension for all 4 main fingers
        index_open = self.check_finger_extended(points, 8, 6, 5)
        middle_open = self.check_finger_extended(points, 12, 10, 9)
        ring_open = self.check_finger_extended(points, 16, 14, 13)
        pinky_open = self.check_finger_extended(points, 20, 18, 17)

        dist_thumb_pinky = self._euclidean_distance(thumb_tip, pinky_mcp)
        thumb_open = dist_thumb_pinky > self._euclidean_distance(thumb_mcp, pinky_mcp) * 0.85

        open_fingers_count = sum([index_open, middle_open, ring_open, pinky_open])

        # 3. 2-Finger Capture Gesture (Index + Middle extended, Ring & Pinky folded, NOT grabbing)
        is_two_fingers = (
            index_open
            and middle_open
            and not ring_open
            and not pinky_open
            and not is_grab
        )

        # 4. OPEN PALM Gesture (All 5 fingers extended and spread apart, NOT grabbing)
        is_open_palm = (
            open_fingers_count >= 4
            and thumb_open
            and not is_grab
        )

        extra_info = {
            "grab_spread": grab_spread,
            "grab_center": grab_center,
            "open_count": open_fingers_count,
            "is_grab": is_grab,
            "is_two_fingers": is_two_fingers,
            "is_open_palm": is_open_palm,
            "index_open": index_open,
            "middle_open": middle_open,
            "thumb_open": thumb_open,
            "ring_open": ring_open,
            "pinky_open": pinky_open,
        }

        # EXCLUSIVE DECISION LOGIC (Strict Hierarchy):
        # Rule 1: GRAB (all 5 tips clustered) has HIGHEST priority
        if is_grab:
            return self.GESTURE_GRAB, extra_info

        # Rule 2: 2-Finger Capture gesture
        if is_two_fingers:
            return self.GESTURE_TWO_FINGERS, extra_info

        # Rule 3: Open Palm gesture (only valid when NOT grabbing)
        if is_open_palm:
            return self.GESTURE_OPEN_PALM, extra_info

        return self.GESTURE_NONE, extra_info

    def detect_single_hand_gesture(self, points):
        """Detects single hand gesture with 3-frame temporal voting filter to eliminate jitter noise."""
        raw_gesture, info = self.detect_single_hand_raw(points)

        # Push raw detection to history buffer
        self.gesture_history.append(raw_gesture)

        # Count occurrences in recent history
        if len(self.gesture_history) >= 2:
            # If GRAB was detected in raw frame, output GRAB immediately for fast drag responsiveness
            if raw_gesture == self.GESTURE_GRAB:
                return self.GESTURE_GRAB, info

            # Otherwise require majority consensus
            counts = {}
            for g in self.gesture_history:
                counts[g] = counts.get(g, 0) + 1

            # Find most frequent gesture in history
            majority_gesture = max(counts, key=counts.get)
            if counts[majority_gesture] >= 2:
                return majority_gesture, info

        return raw_gesture, info

