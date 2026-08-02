import time
import cv2
import numpy as np


class CaptureManager:
    """Mengelola alur state penangkapan gambar penuh (Full Frame) di memori RAM"""

    STATE_STREAMING = "STREAMING"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_CAPTURED = "CAPTURED"

    def __init__(self, countdown_seconds=3.0):
        self.countdown_seconds = countdown_seconds

        self.state = self.STATE_STREAMING
        self.countdown_start_time = 0
        self.remaining_time = countdown_seconds

        self.captured_image = None
        self.flash_start_time = 0
        self.flash_duration = 0.35  # Durasi efek kilat kamera dalam detik

    def start_countdown(self):
        """Memulai timer hitung mundur"""
        if self.state != self.STATE_COUNTDOWN:
            self.state = self.STATE_COUNTDOWN
            self.countdown_start_time = time.time()

    def cancel_countdown(self):
        """Membatalkan countdown jika gestur terputus"""
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

    def trigger_capture(self, raw_frame):
        """Mengambil seluruh bingkai/layar gambar murni (Full Frame) di RAM"""
        self.state = self.STATE_CAPTURED
        self.flash_start_time = time.time()
        self.captured_image = raw_frame.copy()
        print("\n[FULL FRAME CAPTURE] Foto seluruh bingkai berhasil ditangkap dan disimpan di RAM!")

    def retake(self):
        """Kembali ke mode streaming dan menghapus tangkapan foto sebelumnya"""
        self.state = self.STATE_STREAMING
        self.remaining_time = self.countdown_seconds
        self.captured_image = None

    def draw_ui(self, display_frame):
        """Menggambar efek visual UI countdown, kilat kamera, dan petunjuk di layar"""
        h, w = display_frame.shape[:2]

        # 1. Efek Shutter Flash
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
            text_str = str(count_num)
            font_scale = 4.0
            thickness = 8
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)

            # Lingkaran progress di tengah layar
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
                "Tahan Gestur 2 Jari!",
                (cx - 140, cy + 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
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
                "FOTO FULL FRAME TERCAPTURE DI MEMORI!",
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
