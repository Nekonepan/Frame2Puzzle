import cv2
import time
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from capture_manager import CaptureManager
from puzzle_manager import PuzzleManager


def main():
    # 1. Camera Initialization
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    # Camera Hardware Optimization: MJPG Format & Maximum FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 60)

    window_name = "Frame2Puzzle - Gesture Hold Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 2. Initialize Hand Tracker, Gesture Recognizer, Capture Manager, & Puzzle Manager
    print("Initializing Hand Tracker, Gesture Recognizer, Capture Manager, & Puzzle Manager...")
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)
    recognizer = GestureRecognizer(pinch_threshold_px=40)
    capture_mgr = CaptureManager(countdown_seconds=3.0, gesture_hold_seconds=1.2)
    puzzle_mgr = PuzzleManager(rows=3, cols=3)

    STATE_PUZZLE_GAME = "PUZZLE_GAME"

    prev_time = time.time()

    print("\n=======================================================")
    print("   Frame2Puzzle - Phase 4/5: Gesture Hold Capture")
    print("=======================================================")
    print("Instructions:")
    print("1. Hold 2-FINGER GESTURE stably for 1.2 seconds.")
    print("2. A circular loading progress will appear (0% -> 100%).")
    print("3. Once held for 1.2s, the 3-second countdown starts!")
    print("4. Show PINCH gesture (or press SPACE) to start Puzzle Game.")
    print("5. Show OPEN PALM gesture (or press 'r') anytime to Retake.")
    print("6. Press 'q' or 'ESC' to exit.")
    print("=======================================================\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break

        # Horizontal flip for natural mirror view
        frame = cv2.flip(frame, 1)

        # Store clean unannotated raw frame for full-frame puzzle capture
        raw_frame = frame.copy()

        # 3. Hand Landmark Detection & Point Extraction
        results = tracker.process_frame(frame)
        hands_pts = tracker.get_hand_points(frame, results)

        # Detect Single-Hand Gestures (TWO_FINGERS, OPEN_PALM, PINCH)
        is_two_fingers_detected = False
        any_open_palm = False
        any_pinch = False
        pinch_centers = []
        active_gestures = []

        for idx, pts in enumerate(hands_pts):
            gesture_name, info = recognizer.detect_single_hand_gesture(pts)

            if gesture_name == GestureRecognizer.GESTURE_TWO_FINGERS:
                is_two_fingers_detected = True
                active_gestures.append("CAPTURE (2-Fingers)")
            elif gesture_name == GestureRecognizer.GESTURE_OPEN_PALM:
                any_open_palm = True
                active_gestures.append("OPEN PALM")
            elif gesture_name == GestureRecognizer.GESTURE_PINCH:
                any_pinch = True
                active_gestures.append("PINCH")
                pinch_centers.append(info["pinch_center"])

        # 4. State Machine (STREAMING -> COUNTDOWN -> CAPTURED -> PUZZLE_GAME)
        if capture_mgr.state == CaptureManager.STATE_STREAMING:
            # Require holding 2-finger gesture continuously for 1.2 seconds
            hold_complete = capture_mgr.update_gesture_hold(is_two_fingers_detected)
            if hold_complete:
                capture_mgr.start_countdown()

        elif capture_mgr.state == CaptureManager.STATE_COUNTDOWN:
            is_finished = capture_mgr.update_countdown()
            if is_finished:
                capture_mgr.trigger_capture(raw_frame)

        elif capture_mgr.state == CaptureManager.STATE_CAPTURED:
            if any_open_palm:
                capture_mgr.retake()
            elif any_pinch:
                print("\n[GAME STATE] PINCH gesture detected! Generating 3x3 Puzzle...")
                puzzle_mgr.generate_puzzle(capture_mgr.captured_image)
                capture_mgr.state = STATE_PUZZLE_GAME

        elif capture_mgr.state == STATE_PUZZLE_GAME:
            if any_open_palm:
                capture_mgr.retake()

        # 5. Visual Rendering & Overlays
        if capture_mgr.state == STATE_PUZZLE_GAME:
            frame = puzzle_mgr.render_puzzle(frame)
        else:
            if capture_mgr.state == CaptureManager.STATE_CAPTURED and capture_mgr.captured_image is not None:
                preview_img = capture_mgr.captured_image
                ph, pw = preview_img.shape[:2]
                max_preview_size = 220
                scale = min(max_preview_size / pw, max_preview_size / ph)
                nw, nh = int(pw * scale), int(ph * scale)
                resized_preview = cv2.resize(preview_img, (nw, nh))

                fh, fw = frame.shape[:2]
                px1, py1 = fw - nw - 20, fh - nh - 20
                px2, py2 = px1 + nw, py1 + nh
                frame[py1:py2, px1:px2] = resized_preview
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "RAM Photo Preview",
                    (px1, py1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Show PINCH Gesture or Press SPACE to Start Puzzle!",
                    (20, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Draw hand landmarks
        frame = tracker.draw_landmarks(frame, results, is_mirrored=True)

        # Render PINCH Cursor
        for cx, cy in pinch_centers:
            cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 16, (255, 255, 255), 2, cv2.LINE_AA)

        # Render UI Countdown / Flash / Banner / Hold progress
        if capture_mgr.state != STATE_PUZZLE_GAME:
            frame = capture_mgr.draw_ui(frame)

        # 6. Calculate & Render FPS & Gesture Badges
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if active_gestures and capture_mgr.state != STATE_PUZZLE_GAME:
            cv2.putText(
                frame,
                f"Gesture: {' | '.join(active_gestures)}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Display Frame
        cv2.imshow(window_name, frame)

        # Keyboard Shortcuts: SPACE (Start Puzzle), 'r' (Retake), 'q'/ESC (Quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            capture_mgr.retake()
        elif key == 32:  # Spacebar
            if capture_mgr.state == CaptureManager.STATE_CAPTURED:
                puzzle_mgr.generate_puzzle(capture_mgr.captured_image)
                capture_mgr.state = STATE_PUZZLE_GAME

    # Cleanup resources
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
