#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/deploy/var/www/sistema-gestao/cidadesverdes"

if [[ -d "/home/ubuntu/cidadesverdes" ]]; then
  PROJECT_DIR="/home/ubuntu/cidadesverdes"
fi

cd "$PROJECT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export API_PORT="${API_PORT:-8101}"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

exec uvicorn app_ambiente.main:app --host 0.0.0.0 --port "$API_PORT"
