#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 22.12 or newer is required." >&2
  exit 1
fi

npm install
exec npm run dev -- --host "${HOST:-127.0.0.1}" --port "${PORT:-4321}"
