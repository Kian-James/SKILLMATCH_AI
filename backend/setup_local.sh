#!/bin/bash
# One-time local setup for SkillMatch AI backend.
# Run once from the backend/ directory: bash setup_local.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SkillMatch AI Local Setup ==="
echo ""

# Check Python 3.11
if ! command -v python3.11 &>/dev/null; then
    echo "ERROR: python3.11 not found."
    echo "Install from https://www.python.org/downloads/ or:"
    echo "  macOS:  brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
echo "Python 3.11: $(python3.11 --version)"

# Check Tesseract
if ! command -v tesseract &>/dev/null; then
    echo ""
    echo "WARNING: Tesseract not found. Install it:"
    echo "  macOS:  brew install tesseract"
    echo "  Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils"
    echo ""
fi

# Create .env if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example — EDIT IT before running."
    echo "  Set FIREBASE_SERVICE_ACCOUNT_PATH and HUGGINGFACE_API_TOKEN"
    echo ""
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip --quiet

echo "Installing PyTorch CPU..."
pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu --quiet

echo "Installing ML dependencies..."
pip install numpy==1.26.4 --quiet
pip install sentence-transformers==3.0.0 faiss-cpu==1.8.0 scikit-learn==1.4.2 --quiet

echo "Installing spaCy + English model..."
pip install spacy==3.7.4 --quiet
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl --quiet

echo "Installing app requirements..."
pip install -r requirements.txt --quiet
pip install pdfplumber==0.11.0 --quiet

echo ""
echo "=== Setup complete! ==="
echo ""
echo "1. Edit .env and set your Firebase + HuggingFace credentials"
echo "2. Run: ./run_local.sh"
echo "3. Then: open http://localhost:8000/warmup"
echo ""

# Show the user's local IP so they can configure Flutter
echo "Your local IP addresses (for Flutter on physical device):"
if command -v ipconfig &>/dev/null; then
    ipconfig | grep "IPv4" | awk '{print "  " $NF}'
elif command -v hostname &>/dev/null; then
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$' | head -3 | awk '{print "  " $1}'
fi
echo ""
echo "In Flutter, set your base URL to:"
echo "  Android emulator:  http://10.0.2.2:8000"
echo "  iOS simulator:     http://localhost:8000"
echo "  Physical device:   http://<your-ip-above>:8000"
