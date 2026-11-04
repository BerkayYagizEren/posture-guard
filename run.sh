#!/usr/bin/env bash
# Starts Posture Guard from its virtualenv, creating it on first run.
# Arguments are passed through to app.py, e.g. ./run.sh --sound purr
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "Sanal ortam kuruluyor (.venv)..."
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python app.py "$@"
