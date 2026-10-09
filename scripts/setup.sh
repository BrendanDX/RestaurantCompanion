#!/usr/bin/env bash
# One-shot setup: venv + dependencies + .env
# Usage:  bash scripts/setup.sh          (light install: Module 1 + tests)
#         bash scripts/setup.sh --all    (everything, incl. torch/chroma/gradio/mcp)
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

if [[ "${1:-}" == "--all" ]]; then
  pip install -e ".[all,dev]"
else
  pip install -e ".[dev]"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env - open it and paste your DEEPSEEK_API_KEY."
fi

echo
echo "Done. Next:  source .venv/bin/activate && pytest"
