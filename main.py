import cv2
import time
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer


def main():
    # 1. Inisialisasi Kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    window_name = "Frame2Puzzle - Fase 3 (Gesture Recognition)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 2. Inisialisasi Hand Tracker & Gesture Recognizer
    print("Menginisialisasi MediaPipe Hand Tracking & Gesture Recognition...")
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)
    recognizer = GestureRecognizer(pinch_threshold_px=40)

    prev_time = 0

    print("\n--- Fase 3 Aktif: Pengenalan Gestur Tangan ---")
    print("Gestur yang didukung:")
    print("1. PINCH      : Jepit Ujung Ibu Jari (#4) & Ujung Telunjuk (#8)")
    print("2. OPEN PALM  : Buka Kelima Jari Tangan")
    print("3. FRAME      : Bentuk Bingkai Persegi Menggunakan Dua Tangan")
    print("\nTekan 'q' atau 'ESC' pada jendela kamera untuk keluar.\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Gagal mengambil frame dari kamera.")
            break

        # Flip frame horisontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        # 3. Deteksi Landmark Tangan
        results = tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame, results, is_mirrored=True)

        # Dapatkan titik-titik landmark (pixel coordinates)
        hands_pts = tracker.get_hand_points(frame, results)

        active_gestures = []

        # 4. Deteksi Gestur Dua Tangan (FRAME GESTURE)
        is_frame_detected, frame_roi = recognizer.detect_frame_gesture(hands_pts)

        if is_frame_detected and frame_roi:
            active_gestures.append("FRAME GESTURE")
            min_x, min_y, max_x, max_y = frame_roi

            # Gambar Bingkai Foto (Frame ROI) dengan sudut berkilau (Neon Green/Cyan)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)

            # Gambar sudut-sudut dekoratif pada bingkai
            corner_len = 25
            # Top-Left
            cv2.line(frame, (min_x, min_y), (min_x + corner_len, min_y), (0, 255, 255), 5)
            cv2.line(frame, (min_x, min_y), (min_x, min_y + corner_len), (0, 255, 255), 5)
            # Top-Right
            cv2.line(frame, (max_x, min_y), (max_x - corner_len, min_y), (0, 255, 255), 5)
            cv2.line(frame, (max_x, min_y), (max_x, min_y + corner_len), (0, 255, 255), 5)
            # Bottom-Left
            cv2.line(frame, (min_x, max_y), (min_x + corner_len, max_y), (0, 255, 255), 5)
            cv2.line(frame, (min_x, max_y), (min_x, max_y - corner_len), (0, 255, 255), 5)
            # Bottom-Right
            cv2.line(frame, (max_x, max_y), (max_x - corner_len, max_y), (0, 255, 255), 5)
            cv2.line(frame, (max_x, max_y), (max_x, max_y - corner_len), (0, 255, 255), 5)

            # Label Bingkai Foto
            cv2.putText(
                frame,
                "FRAME DETECTED (Siap Tangkap Gambar)",
                (min_x, max(30, min_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 5. Deteksi Gestur Satu Tangan per Tangan (PINCH, OPEN_PALM)
        for idx, pts in enumerate(hands_pts):
            gesture_name, info = recognizer.detect_single_hand_gesture(pts)

            if gesture_name != GestureRecognizer.GESTURE_NONE:
                active_gestures.append(f"Tangan {idx+1}: {gesture_name}")

            # Visual Efek khusus untuk Pinch (Kursor Drag & Drop)
            if gesture_name == GestureRecognizer.GESTURE_PINCH:
                cx, cy = info["pinch_center"]
                cv2.circle(frame, (cx, cy), 14, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    "PINCH / GRAB",
                    (cx + 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        # 6. Menghitung FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Overlay Informasi UI
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Tampilkan Badge Status Gestur Aktif
        if active_gestures:
            gestures_str = " | ".join(active_gestures)
            cv2.putText(
                frame,
                f"Gestur: {gestures_str}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Gestur: Menunggu Gestur...",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        # Menampilkan Frame ke Jendela Aplikasi
        cv2.imshow(window_name, frame)

        # Keluar jika pengguna menekan 'q' atau 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Cleanup resources
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
