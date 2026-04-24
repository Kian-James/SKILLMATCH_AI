#!/bin/bash
# Run the SkillMatch AI backend locally.
# Usage: ./run_local.sh
# Logs stream directly to your terminal in real time.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    # Windows Git Bash
    source venv/Scripts/activate
else
    echo "ERROR: venv not found. Run setup first:"
    echo "  python3.11 -m venv venv && source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "================================"
echo " SkillMatch AI — Local Backend"
echo " http://localhost:8000"
echo " Warmup: http://localhost:8000/warmup"
echo " Docs:   http://localhost:8000/docs"
echo " Ctrl+C to stop"
echo "================================"
echo ""

python main.py
