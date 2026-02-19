#!/bin/bash
# Script to install virtual environment

echo "=== Creating virtual environment ==="
python3 -m venv venv

echo ""
echo "=== Installing dependencies ==="
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo ""
echo "=== Installing demucs submodule ==="
venv/bin/pip install -e submodules/demucs

echo ""
echo "=== Installation complete! ==="
echo "To activate: source venv/bin/activate"
