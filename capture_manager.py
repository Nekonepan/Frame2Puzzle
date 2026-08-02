import time
import cv2
import numpy as np


class CaptureManager:
    """Manages the in-memory photo capture state machine (Gesture Hold -> Countdown -> Shutter Flash -> RAM Store)."""

    STATE_STREAMING = "STREAMING"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_CAPTURED = "CAPTURED"

    def __init__(self, countdown_seconds=3.0, gesture_hold_seconds=1.2):
        self.countdown_seconds = countdown_seconds
        self.gesture_hold_seconds = gesture_hold_seconds

        self.state = self.STATE_STREAMING
        self.countdown_start_time = 0
        self.remaining_time = countdown_seconds

        # Gesture hold tracking
        self.gesture_hold_start_time = 0
        self.current_hold_progress = 0.0  # 0.0 to 1.0

        # In-memory captured image stored as NumPy ndarray (RAM)
        self.captured_image = None
        self.flash_start_time = 0
        self.flash_duration = 0.35  # Shutter flash effect duration in seconds

    def update_gesture_hold(self, is_two_fingers_detected):
        """Tracks continuous gesture hold for 1.2 seconds before starting countdown."""
        if self.state != self.STATE_STREAMING:
            self.gesture_hold_start_time = 0
            self.current_hold_progress = 0.0
            return False

        if is_two_fingers_detected:
            if self.gesture_hold_start_time == 0:
                self.gesture_hold_start_time = time.time()

            elapsed = time.time() - self.gesture_hold_start_time
            self.current_hold_progress = min(1.0, elapsed / self.gesture_hold_seconds)

            if elapsed >= self.gesture_hold_seconds:
                self.gesture_hold_start_time = 0
                self.current_hold_progress = 0.0
                return True  # Hold duration reached!
        else:
            self.gesture_hold_start_time = 0
            self.current_hold_progress = 0.0

        return False

    def start_countdown(self):
        """Starts the 3-second countdown timer."""
        if self.state != self.STATE_COUNTDOWN:
            self.state = self.STATE_COUNTDOWN
            self.countdown_start_time = time.time()

    def cancel_countdown(self):
        """Cancels the countdown if explicitly requested."""
        self.state = self.STATE_STREAMING
        self.remaining_time = self.countdown_seconds

    def update_countdown(self):
        """Updates remaining countdown time. Returns True when countdown reaches 0 seconds."""
        if self.state != self.STATE_COUNTDOWN:
            return False

        elapsed = time.time() - self.countdown_start_time
        self.remaining_time = max(0.0, self.countdown_seconds - elapsed)

        if self.remaining_time <= 0:
            return True
        return False

    def trigger_capture(self, raw_frame):
        """Captures the full raw camera frame into memory (RAM) without writing to disk."""
        self.state = self.STATE_CAPTURED
        self.flash_start_time = time.time()
        self.captured_image = raw_frame.copy()
        print("\n[FULL FRAME CAPTURE] Photo captured and stored in memory (RAM)!")

    def retake(self):
        """Resets back to streaming state and clears previously captured RAM image."""
        self.state = self.STATE_STREAMING
        self.remaining_time = self.countdown_seconds
        self.captured_image = None
        self.gesture_hold_start_time = 0
        self.current_hold_progress = 0.0

    def draw_ui(self, display_frame):
        """Renders visual UI overlays: gesture hold progress, countdown timer, shutter flash, and instructions."""
        h, w = display_frame.shape[:2]

        # 0. Render Gesture Hold Progress Bar in STREAMING state
        if self.state == self.STATE_STREAMING and self.current_hold_progress > 0:
            pct = int(self.current_hold_progress * 100)
            cx, cy = w // 2, h // 2

            # Hold progress circle indicator
            radius = 60
            cv2.circle(display_frame, (cx, cy), radius, (0, 0, 0), -1)
            
            # Progress arc / ring
            angle = int(360 * self.current_hold_progress)
            cv2.ellipse(display_frame, (cx, cy), (radius, radius), 0, -90, -90 + angle, (0, 255, 255), 6)

            cv2.putText(
                display_frame,
                f"{pct}%",
                (cx - 25, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Holding 2-Finger Gesture...",
                (cx - 140, cy + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 1. Shutter Flash Effect (White flash overlay when photo is taken)
        elif self.state == self.STATE_CAPTURED and (time.time() - self.flash_start_time < self.flash_duration):
            flash_overlay = np.full_like(display_frame, 255)
            alpha = 1.0 - ((time.time() - self.flash_start_time) / self.flash_duration)
            cv2.addWeighted(flash_overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        # 2. Countdown Timer UI Rendering
        elif self.state == self.STATE_COUNTDOWN:
            count_num = int(np.ceil(self.remaining_time))
            if count_num < 1:
                count_num = 1

            cx, cy = w // 2, h // 2
            text_str = str(count_num)
            font_scale = 4.0
            thickness = 8
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)

            cv2.circle(display_frame, (cx, cy), 80, (0, 0, 0), -1)
            cv2.circle(display_frame, (cx, cy), 80, (0, 255, 255), 5)

            cv2.putText(
                display_frame,
                text_str,
                (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                "Hold Pose / Get Ready!",
                (cx - 150, cy + 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 3. Captured Photo Preview UI Banner
        elif self.state == self.STATE_CAPTURED:
            banner_h = 70
            cv2.rectangle(display_frame, (0, 0), (w, banner_h), (30, 30, 30), -1)
            cv2.line(display_frame, (0, banner_h), (w, banner_h), (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                "PHOTO CAPTURED IN RAM! Ready for Puzzle Generation.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Show OPEN PALM Gesture or Press 'r' to Retake Photo",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return display_frame
