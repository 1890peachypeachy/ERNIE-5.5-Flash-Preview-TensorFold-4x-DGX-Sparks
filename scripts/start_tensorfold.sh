#!/usr/bin/env bash
# Start ERNIE 5.5 Flash Preview on tensorfold-native, TP=4 over RoCE: workers (ranks 3..1) first, then rank 0 here.
set -euo pipefail
cd "$(dirname "$0")"
source ./cluster.env
bash ./stop_tensorfold.sh || true
sleep 1

ENV_COMMON="LD_LIBRARY_PATH=$ROOT/lib NCCL_SOCKET_IFNAME=$NET_IFACE NCCL_IB_HCA=$HCAS NCCL_IB_GID_INDEX=$GID_INDEX NCCL_IB_TC=$IB_TC NCCL_CROSS_NIC=1 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 TF_ROCE=1 TF_ROCE_HCA=$HCAS TF_ROCE_GID_INDEX=$GID_INDEX"
ARGS_COMMON="$MODEL_DIR --backend cuda --host 0.0.0.0 --port $PORT --name $SERVED_NAME --context $CONTEXT --thinking --tp $((${#WORKERS[@]} + 1)) --master $MASTER_IP --master-port $MASTER_PORT"

mkdir -p "$ROOT/logs"
for ((i = ${#WORKERS[@]}; i >= 1; i--)); do
  h=${WORKERS[$((i - 1))]}
  ssh -o BatchMode=yes "$h" "nohup env $ENV_COMMON $ROOT/bin/tensorfold-native serve $ARGS_COMMON --rank $i >/tmp/tf_ernie_r$i.log 2>&1 </dev/null &"
  echo "launched rank $i on $h"
done
nohup env $ENV_COMMON "$ROOT/bin/tensorfold-native" serve $ARGS_COMMON --rank 0 >"$ROOT/logs/tf_ernie_r0.log" 2>&1 </dev/null &
echo "launched rank 0 here (log: $ROOT/logs/tf_ernie_r0.log)"

echo "waiting for http://127.0.0.1:$PORT/v1/models (first boot ~5 min) ..."
for i in $(seq 1 200); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "READY after $((i * 3))s"; exit 0; fi
  pgrep -f "bin/tensorfold-native" >/dev/null || { echo "rank 0 exited:"; tail -30 "$ROOT/logs/tf_ernie_r0.log"; exit 1; }
  sleep 3
done
echo "timeout; tail of the rank-0 log:"; tail -30 "$ROOT/logs/tf_ernie_r0.log"; exit 1
