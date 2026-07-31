import cv2
import time
from hand_tracker import HandTracker


def main():
    # 1. Inisialisasi Kamera (Resolusi Default Bawaan Webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    window_name = "Frame2Puzzle - Fase 2 (Hand Tracking)"

    # Buat jendela responsif untuk Hyprland (WINDOW_NORMAL)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 2. Inisialisasi MediaPipe Hand Tracker
    print("Menginisialisasi MediaPipe Hand Tracking...")
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)

    # Variabel untuk menghitung FPS
    prev_time = 0
    curr_time = 0

    print("Kamera Aktif dengan Dimensi Default!")
    print("- Hand Tracking 21 Landmarks Aktif (Koreksi Mirror Kanan/Kiri).")
    print("- Tekan 'q' atau 'ESC' pada jendela kamera untuk keluar.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Gagal mengambil frame dari kamera.")
            break

        # Flip frame horisontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        # 3. Proses Deteksi Tangan & 21 Landmarks Real-Time
        results = tracker.process_frame(frame)

        # 4. Visualisasi Landmark Tangan pada Frame Kamera
        frame = tracker.draw_landmarks(frame, results, is_mirrored=True)

        # Hitung jumlah tangan yang terdeteksi
        num_hands_detected = (
            len(results.hand_landmarks) if results.hand_landmarks else 0
        )

        # 5. Menghitung FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Menampilkan Teks Overlay (FPS & Jumlah Tangan)
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
        cv2.putText(
            frame,
            f"Tangan Terdeteksi: {num_hands_detected}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0) if num_hands_detected > 0 else (200, 200, 200),
            2,
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
