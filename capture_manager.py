import time
import cv2
import numpy as np


class CaptureManager:
    """Mengelola alur state penangkapan gambar: Countdown -> Shutter Flash -> In-Memory Crop (RAM)"""

    STATE_STREAMING = "STREAMING"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_CAPTURED = "CAPTURED"

    def __init__(self, countdown_seconds=3.0):
        self.countdown_seconds = countdown_seconds

        self.state = self.STATE_STREAMING
        self.countdown_start_time = 0
        self.remaining_time = countdown_seconds

        # Gambar hasil tangkapan disimpan murni di memori RAM (NumPy ndarray)
        self.captured_image = None
        self.cropped_roi_image = None
        self.flash_start_time = 0
        self.flash_duration = 0.35  # Durasi efek kilat kamera dalam detik

    def start_countdown(self):
        """Memulai timer hitung mundur"""
        if self.state != self.STATE_COUNTDOWN:
            self.state = self.STATE_COUNTDOWN
            self.countdown_start_time = time.time()

    def cancel_countdown(self):
        """Membatalkan countdown jika gestur bingkai terputus"""
        self.state = self.STATE_STREAMING
        self.remaining_time = self.countdown_seconds

    def update_countdown(self):
        """Mengelaborasi sisa waktu countdown. Mengembalikan True jika waktu habis (0 detik)."""
        if self.state != self.STATE_COUNTDOWN:
            return False

        elapsed = time.time() - self.countdown_start_time
        self.remaining_time = max(0.0, self.countdown_seconds - elapsed)

        if self.remaining_time <= 0:
            return True
        return False

    def trigger_capture(self, raw_frame, frame_roi=None):
        """Mengambil dan memotong gambar murni di dalam memori RAM (tanpa menulis ke disk)"""
        self.state = self.STATE_CAPTURED
        self.flash_start_time = time.time()

        h, w = raw_frame.shape[:2]

        if frame_roi:
            min_x, min_y, max_x, max_y = frame_roi
            # Batasi koordinat agar berada dalam rentang gambar
            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(w, max_x), min(h, max_y)

            # Jika area potong cukup besar (> 80px), gunakan area bingkai
            if (max_x - min_x) > 80 and (max_y - min_y) > 80:
                self.cropped_roi_image = raw_frame[min_y:max_y, min_x:max_x].copy()
            else:
                self.cropped_roi_image = raw_frame.copy()
        else:
            self.cropped_roi_image = raw_frame.copy()

        self.captured_image = raw_frame.copy()
        print("\n[RAM CAPTURE] Gambar berhasil ditangkap dan disimpan di dalam memori (RAM)!")

    def retake(self):
        """Kembali ke mode streaming dan menghapus memori tangkapan foto sebelumnya"""
        self.state = self.STATE_STREAMING
        self.remaining_time = self.countdown_seconds
        self.captured_image = None
        self.cropped_roi_image = None

    def draw_ui(self, display_frame, frame_roi=None):
        """Menggambar efek visual UI countdown, kilat kamera, dan petunjuk di layar"""
        h, w = display_frame.shape[:2]

        # 1. Efek Shutter Flash (Layar berkedip putih saat mengambil foto)
        if self.state == self.STATE_CAPTURED and (time.time() - self.flash_start_time < self.flash_duration):
            flash_overlay = np.full_like(display_frame, 255)
            alpha = 1.0 - ((time.time() - self.flash_start_time) / self.flash_duration)
            cv2.addWeighted(flash_overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        # 2. Rendering UI Hitung Mundur (COUNTDOWN)
        elif self.state == self.STATE_COUNTDOWN:
            count_num = int(np.ceil(self.remaining_time))
            if count_num < 1:
                count_num = 1

            cx, cy = w // 2, h // 2
            if frame_roi:
                min_x, min_y, max_x, max_y = frame_roi
                cx = (min_x + max_x) // 2
                cy = (min_y + max_y) // 2

            text_str = str(count_num)
            font_scale = 3.5
            thickness = 7
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)

            cv2.circle(display_frame, (cx, cy), 70, (0, 0, 0), -1)
            cv2.circle(display_frame, (cx, cy), 70, (0, 255, 255), 4)

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
                "Tahan Bingkai Tangan!",
                (cx - 130, cy + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 3. Rendering UI Foto Berhasil Ditangkap (CAPTURED PREVIEW)
        elif self.state == self.STATE_CAPTURED:
            banner_h = 70
            cv2.rectangle(display_frame, (0, 0), (w, banner_h), (30, 30, 30), -1)
            cv2.line(display_frame, (0, banner_h), (w, banner_h), (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                "FOTO TERCAPTURE DI MEMORI! siap dijadikan puzzle.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Buka Telapak Tangan (Open Palm) atau Tekan 'r' untuk Foto Ulang",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return display_frame
