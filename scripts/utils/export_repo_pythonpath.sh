#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
