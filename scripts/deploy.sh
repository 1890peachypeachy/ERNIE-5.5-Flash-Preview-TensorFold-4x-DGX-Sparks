#!/usr/bin/env bash
# Stop the lane, wait until no rank holds the binary, back up the old one, copy the new build to every rank
# (verify md5), start. Run on rank 0 after `zig build native`.
set -euo pipefail
cd "$(dirname "$0")"
source ./cluster.env
TF=${TF:-$ROOT/tensorfold-ernie}
bash ./stop_tensorfold.sh
count() { pgrep -fc "^$ROOT/bin/tensorfold-native" || true; }
for i in $(seq 1 30); do
  n=$(count); for h in "${WORKERS[@]}"; do n=$((n + $(ssh "$h" "pgrep -fc '^$ROOT/bin/tensorfold-native' || true"))); done
  [ "$n" -eq 0 ] && break; sleep 2
done
mkdir -p "$ROOT/backups"
[ -f "$ROOT/bin/tensorfold-native" ] && cp "$ROOT/bin/tensorfold-native" "$ROOT/backups/tensorfold-native-$(date +%Y%m%dT%H%M)"
cp "$TF/zig-out/native/bin/tensorfold-native" "$ROOT/bin/tensorfold-native"
for h in "${WORKERS[@]}"; do scp -q "$ROOT/bin/tensorfold-native" "$h:$ROOT/bin/tensorfold-native" & done; wait
md5sum "$ROOT/bin/tensorfold-native" | cut -c1-12
for h in "${WORKERS[@]}"; do ssh "$h" md5sum "$ROOT/bin/tensorfold-native" | cut -c1-12; done
bash ./start_tensorfold.sh
