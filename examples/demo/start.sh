#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
exec "${GUARD_PYTHON:-python}" -u server.py \
  --bundle "${GUARD_MODEL_DIR:-./models/HS-Guard-v10}" \
  --host "${GUARD_HOST:-127.0.0.1}" --port "${GUARD_PORT:-8080}" --static ./static
