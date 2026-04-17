#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

cd "$PROJECT_ROOT"

if command -v uv >/dev/null 2>&1; then
  uv sync
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3.10 or newer, then run this script again." >&2
  exit 1
fi

VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools >/dev/null
python -m pip install -e .
