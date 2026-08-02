VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all run setup clean help

# Target default ketika mengetik 'make'
all: run

# Menjalankan aplikasi Frame2Puzzle
run: $(VENV)
	@echo "🚀 Menjalankan Frame2Puzzle..."
	@$(PYTHON) main.py

# Membentuk virtual environment & menginstal dependensi dari requirements.txt
setup: $(VENV)

$(VENV): requirements.txt
	@echo "📦 Membentuk Virtual Environment (.venv) & Menginstal Dependensi..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup selesai!"

# Membersihkan cache Python dan virtual environment
clean:
	@echo "🧹 Membersihkan file cache dan virtual environment..."
	rm -rf $(VENV)
	rm -rf __pycache__ *.pyc
	rm -rf .pytest_cache
	@echo "✨ Bersih!"

# Menampilkan daftar perintah yang tersedia
help:
	@echo "======================================================="
	@echo "            Makefile Frame2Puzzle Project              "
	@echo "======================================================="
	@echo "  make         : Jalankan aplikasi Frame2Puzzle secara langsung"
	@echo "  make run     : Menjalankan aplikasi Frame2Puzzle"
	@echo "  make setup   : Membuat .venv & memasang seluruh dependensi"
	@echo "  make clean   : Menghapus .venv dan file cache Python"
	@echo "  make help    : Menampilkan bantuan perintah ini"
	@echo "======================================================="
