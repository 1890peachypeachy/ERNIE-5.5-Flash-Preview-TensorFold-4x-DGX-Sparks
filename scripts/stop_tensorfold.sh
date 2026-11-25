#!/usr/bin/env bash
# Stop tensorfold-native on every rank. Matches the binary path exactly (a bare `pkill -f tensorfold` over ssh can
# match the ssh command line itself and kill the session).
set -uo pipefail
cd "$(dirname "$0")"
source ./cluster.env
pkill -f "^$ROOT/bin/tensorfold-native" || true
for h in "${WORKERS[@]}"; do ssh -o BatchMode=yes "$h" "pkill -f '^$ROOT/bin/tensorfold-native' || true" & done
wait
echo "stopped tensorfold-native on all ranks"
