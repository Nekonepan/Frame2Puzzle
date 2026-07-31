import cv2
import time

# HandTracker is optional — fall back gracefully if module or model not present
try:
    from hand_tracker import HandTracker
except Exception:
    HandTracker = None


def aspect_fill_crop(frame, target_w, target_h):
    """Melakukan Center-Crop (Aspect Fill / object-fit: cover) pada frame

    agar memenuhi 100% lebar dan tinggi jendela tanpa sisa hitam
    dan tanpa membuat gambar terdistorsi/gepeng.
    """
    fh, fw = frame.shape[:2]
    if target_w <= 0 or target_h <= 0 or fw <= 0 or fh <= 0:
        return frame

    target_aspect = target_w / target_h
    frame_aspect = fw / fh

    if target_aspect > frame_aspect:
        # Jendela lebih lebar daripada frame kamera -> potong atas & bawah
        new_h = int(fw / target_aspect)
        y1 = max(0, (fh - new_h) // 2)
        y2 = y1 + new_h
        cropped = frame[y1:y2, 0:fw]
    else:
        # Jendela lebih tinggi daripada frame kamera -> potong kiri & kanan
        new_w = int(fh * target_aspect)
        x1 = max(0, (fw - new_w) // 2)
        x2 = x1 + new_w
        cropped = frame[0:fh, x1:x2]

    # Resize hasil potongan ke ukuran tepat jendela
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def main():
    # 1. Inisialisasi Kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    # Minta resolusi kamera HD jika didukung
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Frame2Puzzle - Fase 2 (Hand Tracking)"

    # Buat jendela responsif
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Bebaskan aspek rasio bawaan OpenCV agar tidak memaksa black-bars (letterboxing/pillarboxing)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO
    )

    # 2. Inisialisasi MediaPipe Hand Tracker (jika tersedia)
    if HandTracker is not None:
        print("Menginisialisasi MediaPipe Hand Tracking...")
        tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)
    else:
        tracker = None

    # Variabel untuk menghitung FPS
    prev_time = 0
    curr_time = 0

    if tracker is not None:
        print("Fase 2 Aktif: Deteksi Tangan & 21 Landmarks Real-Time!")
        print("- Arahkan tangan Anda ke kamera untuk melihat 21 titik landmark.")
    else:
        print("Kamera aktif! HandTracker tidak tersedia; menampilkan frame saja.")

    print("- Tekan 'q' atau 'ESC' pada jendela kamera untuk keluar.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Gagal mengambil frame dari kamera.")
            break

        # Flip frame horisontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        # Proses deteksi tangan jika tracker ada
        num_hands_detected = 0
        if tracker is not None:
            results = tracker.process_frame(frame)
            frame = tracker.draw_landmarks(frame, results)
            num_hands_detected = (
                len(results.hand_landmarks) if getattr(results, "hand_landmarks", None) else 0
            )

        # Menghitung FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Dapatkan dimensi jendela untuk auto-resize Aspect Fill
        rect = cv2.getWindowImageRect(window_name)
        if rect is not None and len(rect) == 4:
            _, _, win_w, win_h = rect
            if win_w > 0 and win_h > 0:
                frame = aspect_fill_crop(frame, win_w, win_h)

        # Menampilkan Teks Overlay (FPS & Jumlah Tangan)
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Tangan Terdeteksi: {num_hands_detected}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
    if tracker is not None:
        try:
            tracker.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
