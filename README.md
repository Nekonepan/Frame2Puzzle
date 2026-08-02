# Frame2Puzzle 🧩✋

Frame2Puzzle is a Python-based application leveraging Computer Vision and Real-Time Hand Tracking technology. It allows users to capture images using hand gestures, which are then converted into an interactive puzzle game solvable entirely via hand gestures in real-time—without requiring a mouse or keyboard.

---

## 🚀 Key Features

1. **High Performance Camera Stream & Real-Time FPS**: Optimized OpenCV MJPG camera pipeline running at maximum hardware frame rates.
2. **21-Landmark Hand Tracking**: Uses MediaPipe Hand Landmarker for real-time skeleton tracking and 3D joint landmark detection.
3. **Gesture Recognition Engine**:
   - **`TWO_FINGERS` (Victory / Peace Sign)**: Triggers a 3-second countdown to capture a full-frame photo into memory (RAM).
   - **`OPEN_PALM`**: Retakes the photo and resets the capture state.
   - **`PINCH`**: Interactive cursor for grabbing and moving puzzle pieces.
4. **In-Memory Capture System**: Zero-disk I/O photo capture storing images directly in RAM for speed and privacy.
5. **Hyprland & Tiling WM Friendly**: Native support for Linux Wayland / Tiling Window Managers.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV (`opencv-python`)
- **Hand Tracking**: MediaPipe (`mediapipe`)
- **Array & Image Processing**: NumPy & Pillow

---

## 📥 Quick Start Guide

### Prerequisites
Make sure `python3` and `make` are installed on your system.

### Running the Application
Simply execute GNU Make in your terminal:

```bash
make
```

Or step-by-step:

```bash
# 1. Setup environment and dependencies
make setup

# 2. Run the application
make run
```

---

## 🎮 How to Use

1. **Start the app**: Execute `make run`.
2. **Capture Photo**: Show a **2-Finger Gesture** (Index + Middle finger extended) to start the 3-second countdown.
3. **Pose & Get Ready**: The countdown (3... 2... 1...) runs continuously even if you lower your hand.
4. **Photo Captured**: At 0 seconds, a white flash shutter effect triggers, and the full camera frame is captured in RAM.
5. **Retake**: Show an **Open Palm** gesture (or press `r`) to retake the photo.
6. **Exit**: Press `q` or `ESC` to quit.

---

## 📜 Makefile Commands

| Command | Description |
| :--- | :--- |
| `make` / `make run` | Runs the Frame2Puzzle application |
| `make setup` | Creates `.venv` and installs all dependencies |
| `make clean` | Removes `.venv` and Python cache files |
| `make help` | Displays the Makefile help menu |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).