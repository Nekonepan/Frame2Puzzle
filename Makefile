VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all run setup clean help get-model

# Default target when typing 'make'
all: run

# Run Frame2Puzzle application
run: $(VENV)
	@echo "🚀 Running Frame2Puzzle..."
	@$(PYTHON) main.py

# Download model asset (hand_landmarker.task) into repo root
get-model:
	@echo "⬇️  Fetching hand_landmarker.task model..."
	@./scripts/get_model.sh

# Create virtual environment & install dependencies from requirements.txt
setup: $(VENV)

$(VENV): requirements.txt
	@echo "📦 Creating Virtual Environment (.venv) & Installing Dependencies..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup complete!"

# Clean Python cache and virtual environment
clean:
	@echo "🧹 Cleaning cache files and virtual environment..."
	rm -rf $(VENV)
	rm -rf __pycache__ *.pyc
	rm -rf .pytest_cache
	@echo "✨ Cleaned successfully!"

# Show Makefile help menu
help:
	@echo "======================================================="
	@echo "            Makefile Frame2Puzzle Project              "
	@echo "======================================================="
	@echo "  make         : Run Frame2Puzzle application directly"
	@echo "  make run     : Run Frame2Puzzle application"
	@echo "  make get-model: Download hand_landmarker.task into repo root"
	@echo "  make setup   : Create .venv & install all dependencies"
	@echo "  make clean   : Remove .venv and Python cache files"
	@echo "  make help    : Show this help menu"
	@echo "======================================================="
