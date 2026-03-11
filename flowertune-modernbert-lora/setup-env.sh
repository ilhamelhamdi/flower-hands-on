#!/bin/bash
# setup-env.sh

export PYTHONUSERBASE=$(pwd)/.local
pip install --user uv --break-system-packages
export PATH=$PATH:$(pwd)/.local/bin

if [ ! -d ".venv" ]; then
    uv venv .venv
fi

source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install -e .