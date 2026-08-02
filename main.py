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

    window_name = "Frame2Puzzle - Fase 4 (In-Memory Image Capture)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 2. Inisialisasi Tracker, Recognizer, dan Capture Manager
    print("Menginisialisasi Hand Tracker, Gesture Recognizer, & Capture Manager...")
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)
    recognizer = GestureRecognizer(pinch_threshold_px=40)
    capture_mgr = CaptureManager(countdown_seconds=3.0)

    prev_time = 0

    print("\n=======================================================")
    print("  Fase 4 Aktif: Penangkapan Gambar Temporer (RAM Only)")
    print("=======================================================")
    print("Petunjuk Penggunaan:")
    print("1. Bentuk FRAME GESTURE (2 Tangan Sudut L) di depan kamera.")
    print("2. Tahan gestur bingkai selama 3 detik (Hitung Mundur 3... 2... 1...).")
    print("3. Foto area bingkai akan otomatis ditangkap di RAM saat hitungan 0!")
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

        # Simpan copy mentah bersih tanpa garis/landmark UI untuk foto puzzle
        raw_frame = frame.copy()

        # 3. Deteksi Landmark Tangan & Point Extraction
        results = tracker.process_frame(frame)
        hands_pts = tracker.get_hand_points(frame, results)

        # Deteksi Gestur Bingkai Foto (FRAME GESTURE)
        is_frame_detected, frame_roi = recognizer.detect_frame_gesture(hands_pts)

        # Deteksi Gestur Tangan Tunggal (OPEN_PALM & PINCH)
        any_open_palm = False
        for pts in hands_pts:
            gesture_name, _ = recognizer.detect_single_hand_gesture(pts)
            if gesture_name == GestureRecognizer.GESTURE_OPEN_PALM:
                any_open_palm = True

        # 4. State Machine Alur Capture (STREAMING -> COUNTDOWN -> CAPTURED)
        if capture_mgr.state == CaptureManager.STATE_STREAMING:
            if is_frame_detected:
                capture_mgr.start_countdown()

        elif capture_mgr.state == CaptureManager.STATE_COUNTDOWN:
            if is_frame_detected:
                is_finished = capture_mgr.update_countdown()
                if is_finished:
                    # Ambil foto mentah bersih di area bingkai ROI (Simpan di RAM)
                    capture_mgr.trigger_capture(raw_frame, frame_roi)
            else:
                # Batalkan hitung mundur jika gestur bingkai terputus sebelum 0 detik
                capture_mgr.cancel_countdown()

        elif capture_mgr.state == CaptureManager.STATE_CAPTURED:
            if any_open_palm:
                capture_mgr.retake()

        # 5. Visualisasi Landmarks & UI Overlay
        if capture_mgr.state == CaptureManager.STATE_CAPTURED and capture_mgr.cropped_roi_image is not None:
            # Tampilkan preview foto tangkapan memori di pojok kanan bawah
            preview_img = capture_mgr.cropped_roi_image
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
                "Preview RAM Puzzle",
                (px1, py1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw hand landmarks
        frame = tracker.draw_landmarks(frame, results, is_mirrored=True)

        # Draw Bingkai Foto jika terdeteksi
        if is_frame_detected and frame_roi:
            min_x, min_y, max_x, max_y = frame_roi
            border_color = (0, 255, 0) if capture_mgr.state == CaptureManager.STATE_COUNTDOWN else (0, 255, 255)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), border_color, 3)

            clen = 25
            cv2.line(frame, (min_x, min_y), (min_x + clen, min_y), (0, 255, 255), 5)
            cv2.line(frame, (min_x, min_y), (min_x, min_y + clen), (0, 255, 255), 5)
            cv2.line(frame, (max_x, min_y), (max_x - clen, min_y), (0, 255, 255), 5)
            cv2.line(frame, (max_x, min_y), (max_x, min_y + clen), (0, 255, 255), 5)

        # Render UI Countdown / Flash / Banner
        frame = capture_mgr.draw_ui(frame, frame_roi)

        # 6. Menghitung & Menampilkan FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
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
