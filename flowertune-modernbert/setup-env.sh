#!/bin/bash
# setup-env.sh

echo "Setting up the environment..."

echo "Installing uv..."
export PYTHONUSERBASE=$(pwd)/.local
pip install --user uv --break-system-packages
export PATH=$PATH:$(pwd)/.local/bin

if [ -d ".venv" ]; then
    rm -rf .venv
fi

echo "Creating virtual environment..."
uv venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -r pyproject.toml
uv pip install -e .

echo "Environment setup complete."