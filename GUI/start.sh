#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

env_file=".env"
example_file=".env.example"

if [[ ! -f "$env_file" && -f "$example_file" ]]; then
  cp "$example_file" "$env_file"
  echo "Created .env from template."
fi

if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi

host="${HOST:-127.0.0.1}"
port="${PORT:-8082}"

if [[ ! -d client/dist ]]; then
  echo "client/dist missing - building the web client (one-time)..."
  (cd client && npm install && npm run build)
fi

cat <<TEXT

Starting NeuroGolf Lab (local mode - no external uploads)
  URL: http://${host}:${port}

Exports are validated with the official competition semantics and saved to
The_400_Tasks/Task_by_Task/<task>/prototypes/ in the parent project.

Headless agent examples (from another shell):
  python3 scripts/agent_export.py --task task276 --color-remap '{"6":2}'
  python3 scripts/agent_export.py --task task010 --graph graph.json --validate-only

TEXT

exec python3 launch.py --host "$host" --port "$port"
