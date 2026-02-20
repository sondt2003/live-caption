#!/bin/bash

# setup_envs.sh - Initialize isolated virtual environments for TTS providers
set -e

PROJECT_ROOT=$(pwd)
ENVS_DIR="$PROJECT_ROOT/envs"
mkdir -p "$ENVS_DIR"

echo "🚀 Starting multi-venv setup..."

# 1. Setup XTTS Environment
echo "📦 Setting up XTTS Environment (envs/venv_xtts)..."
python3.12 -m venv "$ENVS_DIR/venv_xtts"
source "$ENVS_DIR/venv_xtts/bin/activate"
pip install --upgrade pip
# Use stable versions that don't conflict with legacy deep learning libs
pip install "numpy<2.0,>=1.25.2" "torch==2.4.1" "transformers<=4.46.2,>=4.43.0"
pip install coqui-tts==0.25.3
deactivate

# 2. Setup VieNeu Environment
echo "📦 Setting up VieNeu Environment (envs/venv_vieneu)..."
python3.12 -m venv "$ENVS_DIR/venv_vieneu"
source "$ENVS_DIR/venv_vieneu/bin/activate"
pip install --upgrade pip
# VieNeu uses newer stack, we let it manage its own llama-cpp and torch
pip install vieneu
deactivate

echo "✅ Multi-venv setup complete!"
echo "Main environment: venv"
echo "XTTS environment: envs/venv_xtts"
echo "VieNeu environment: envs/venv_vieneu"
