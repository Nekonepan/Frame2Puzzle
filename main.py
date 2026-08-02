import cv2
import time
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from capture_manager import CaptureManager


def main():
    # 1. Inisialisasi Kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    # Optimasi Format Piksel MJPG & High FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 60)

    window_name = "Frame2Puzzle - Fase 4 (2-Finger Continuous Countdown)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 2. Inisialisasi Tracker, Recognizer, dan Capture Manager
    print("Menginisialisasi Hand Tracker, Gesture Recognizer, & Capture Manager...")
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)
    recognizer = GestureRecognizer(pinch_threshold_px=40)
    capture_mgr = CaptureManager(countdown_seconds=3.0)

    prev_time = time.time()

    print("\n=======================================================")
    print("   Fase 4: Continuous Countdown Capture (2 Jari)")
    print("=======================================================")
    print("Petunjuk Penggunaan:")
    print("1. Tunjukkan GESTUR 2 JARI (Telunjuk & Tengah) untuk memicu timer.")
    print("2. Sekali terpicu, hitung mundur (3... 2... 1...) akan TERUS BERJALAN")
    print("   meskipun tangan Anda diturunkan!")
    print("3. Seluruh foto layar akan otomatis ditangkap di RAM saat hitungan 0.")
    print("4. Setelah foto ditangkap:")
    print("   - Buka Telapak Tangan (OPEN PALM) atau tekan 'r' untuk Foto Ulang.")
    print("5. Tekan 'q' atau 'ESC' untuk keluar.")
    print("=======================================================\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Gagal mengambil frame dari kamera.")
            break

        # Flip frame horisontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        # Simpan copy mentah bersih tanpa landmark/UI untuk foto puzzle full-frame
        raw_frame = frame.copy()

        # 3. Deteksi Landmark Tangan & Point Extraction
        results = tracker.process_frame(frame)
        hands_pts = tracker.get_hand_points(frame, results)

        # Deteksi Gestur Tangan (TWO_FINGERS, OPEN_PALM, PINCH)
        is_two_fingers_detected = False
        any_open_palm = False
        active_gestures = []

        for idx, pts in enumerate(hands_pts):
            gesture_name, info = recognizer.detect_single_hand_gesture(pts)

            if gesture_name == GestureRecognizer.GESTURE_TWO_FINGERS:
                is_two_fingers_detected = True
                active_gestures.append("CAPTURE (2-Jari)")
            elif gesture_name == GestureRecognizer.GESTURE_OPEN_PALM:
                any_open_palm = True
                active_gestures.append("OPEN PALM")
            elif gesture_name == GestureRecognizer.GESTURE_PINCH:
                active_gestures.append("PINCH")
                cx, cy = info["pinch_center"]
                cv2.circle(frame, (cx, cy), 14, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 3, cv2.LINE_AA)

        # 4. State Machine Alur Capture (Continuous Countdown)
        if capture_mgr.state == CaptureManager.STATE_STREAMING:
            if is_two_fingers_detected:
                capture_mgr.start_countdown()

        elif capture_mgr.state == CaptureManager.STATE_COUNTDOWN:
            # Countdown berjalan terus sampai 0 detik tanpa bergantung pada keaktifan gestur
            is_finished = capture_mgr.update_countdown()
            if is_finished:
                capture_mgr.trigger_capture(raw_frame)

        elif capture_mgr.state == CaptureManager.STATE_CAPTURED:
            if any_open_palm:
                capture_mgr.retake()

        # 5. Visualisasi Landmarks & UI Overlay
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
                "Foto Full-Frame (RAM)",
                (px1, py1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw hand landmarks
        frame = tracker.draw_landmarks(frame, results, is_mirrored=True)

        # Render UI Countdown / Flash / Banner
        frame = capture_mgr.draw_ui(frame)

        # 6. Menghitung & Menampilkan FPS & Status Gestur
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

        if active_gestures:
            cv2.putText(
                frame,
                f"Gestur: {' | '.join(active_gestures)}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Menampilkan Frame ke Jendela Aplikasi
        cv2.imshow(window_name, frame)

        # Keyboard Shortcut: 'r' (Retake), 'q'/ESC (Quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            capture_mgr.retake()

    # Cleanup resources
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
