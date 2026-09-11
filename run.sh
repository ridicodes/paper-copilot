#!/usr/bin/env bash
set -euo pipefail

PORT="${PAPER_COPILOT_PORT:-8501}"
python -m streamlit run app.py --server.port "$PORT"
