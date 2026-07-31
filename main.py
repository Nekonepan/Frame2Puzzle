import cv2
import time


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
    # 1. Inisialisasi Kamera (0 adalah ID default untuk webcam internal/bawaan)
    cap = cv2.VideoCapture(0)

    # Periksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    # Minta resolusi kamera HD jika didukung
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Frame2Puzzle - Fase 1"

    # Buat jendela responsif
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Bebaskan aspek rasio bawaan OpenCV agar tidak memaksa black-bars (letterboxing/pillarboxing)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO
    )

    # Variabel untuk menghitung FPS
    prev_time = 0
    curr_time = 0

    print("Kamera aktif!")
    print(
        "- Mode Aspect-Fill (Cover): Kamera memenuhi 100% lebar & tinggi jendela Hyprland tanpa garis hitam."
    )
    print("- Tekan 'q' atau 'ESC' pada jendela kamera untuk keluar.")

    while True:
        # Membaca frame demi frame dari kamera
        success, frame = cap.read()
        if not success:
            print("Error: Gagal mengambil frame dari kamera.")
            break

        # Flip frame horisontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        # 2. Menghitung FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Dapatkan dimensi aktual dari jendela OpenCV di Hyprland
        rect = cv2.getWindowImageRect(window_name)
        if rect is not None and len(rect) == 4:
            _, _, win_w, win_h = rect
            if win_w > 0 and win_h > 0:
                # Potong & skala frame secara presisi sesuai ukuran jendela (Aspect Fill / Object Fit Cover)
                frame = aspect_fill_crop(frame, win_w, win_h)

        # Menampilkan Teks FPS pada Frame Kamera
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),  # Warna Hijau
            2,
            cv2.LINE_AA,
        )

        # Menampilkan Frame ke Jendela Aplikasi
        cv2.imshow(window_name, frame)

        # Keluar jika pengguna menekan tombol 'q' (ASCII 113) atau 'ESC' (ASCII 27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Membersihkan resource kamera dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
