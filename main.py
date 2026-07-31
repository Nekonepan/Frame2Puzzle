import cv2
import time


def main():
    # 1. Inisialisasi Kamera (0 adalah ID default untuk webcam internal/bawaan)
    cap = cv2.VideoCapture(0)

    # Periksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    # Variabel untuk menghitung FPS
    prev_time = 0
    curr_time = 0

    print("Kamera aktif! Tekan 'q' atau 'ESC' pada jendela kamera untuk keluar.")

    while True:
        # Membaca frame demi frame dari kamera
        success, frame = cap.read()
        if not success:
            print("Error: Gagal mengambil frame dari kamera.")
            break

        # 2. Menghitung FPS (Frames Per Second)
        curr_time = time.time()
        # Selisih waktu antar frame (1 / delta_time)
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Menampilkan Teks FPS pada Frame Kamera
        # Param: image, text, position (x, y), font, scale, color (BGR), thickness
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # Warna Hijau
            2,
        )

        # Menampilkan Frame ke Jendela Aplikasi
        cv2.imshow("Frame2Puzzle - Fase 1", frame)

        # Keluar jika pengguna menekan tombol 'q' (ASCII 113) atau 'ESC' (ASCII 27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Membersihkan resource kamera dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
