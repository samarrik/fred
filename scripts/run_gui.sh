#!/bin/bash
# Run the Fred Gradio GUI

cd "$(dirname "$0")/.."

echo "Starting Fred GUI..."
echo "Open http://localhost:7860 in your browser"
echo ""

python app/gui.py
