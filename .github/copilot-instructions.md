Repository: Frame2Puzzle — a small Python OpenCV-based prototype for capturing and displaying webcam frames.

1) Build / install / run
- Create a venv (recommended):
  python -m venv .venv
  source .venv/bin/activate
- Install dependencies (exact pinning not provided):
  pip install -r requirements.txt
- Run the app (single script):
  python main.py
- Change camera device: edit the cv2.VideoCapture(...) argument in main.py (default 0).

Notes: there is no packaging, build system, test runner, or lint config in this repo. No CI workflows detected.

2) Tests / single-test guidance
- No tests currently present. When tests are added using pytest, run a single test with:
  pytest path/to/test_file.py::test_name

3) High-level architecture (big picture)
- main.py: single-entry script. Opens webcam (OpenCV), computes/overlays FPS, and displays frames in an OpenCV window until 'q' or ESC is pressed.
- requirements.txt: opencv-python, mediapipe, numpy, pillow — mediapipe/pillow are included but not used by main.py; they indicate planned CV/ML/image operations.
- .venv: local virtualenv is commonly included for development and is present in the repository root (ignored by .gitignore normally).

4) Key conventions and repo-specific notes
- Language / comments: code and comments use Indonesian; keep context in mind when generating new code or messages.
- Window behavior: UI is an OpenCV window titled "Frame2Puzzle - Fase 1". Exiting uses 'q' or ESC.
- Camera assumptions: the app expects a desktop environment with an attached camera and a display server (X/Wayland). Running headless (without a display) will not show the window.
- Minimal scope: this repo is a prototype/demo script rather than a packaged library; contributions that add modules should add a test runner and a lightweight CI workflow.
- Dependency list is lightweight and binary (opencv, mediapipe). For reproducible builds add pins (requirements.txt.lock or pyproject + poetry/pip-tools).

5) Existing docs and AI assistant files
- README.md contains only a project title — no usage or developer instructions. No CONTRIBUTING.md, CLAUDE.md, AGENTS.md, or other AI-assistant rules were found. If adding assistant-specific rules (CLAUDE.md, AGENTS.md, .cursorrules, etc.), include a short summary and a pointer here.

6) Suggested additions (for developer/assistant friendliness)
- Add a brief README usage section (run steps above).
- Add a simple pytest test that imports main and checks that main() constructs VideoCapture in a mock-friendly way.
- Add a dev-only requirements-dev.txt or pyproject.toml with pinned deps and linters (ruff/flake8) if stricter checks are desired.

Created file: .github/copilot-instructions.md

If you'd like, update this file to include any repository-specific workflows (packaging, CI, test names) or request adding basic tests and a CI job; otherwise indicate if the current content should be adjusted.