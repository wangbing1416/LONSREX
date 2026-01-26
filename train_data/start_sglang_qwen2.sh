#!/bin/bash

# Read environment variables with defaults
MASTER_ADDR="${LWS_LEADER_ADDRESS:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${LWS_WORKER_INDEX:-0}"
NNODES="${LWS_GROUP_SIZE:-1}"
GPU=4

# MODEL from first argument or environment
MODEL="${1:-YOURMODELPATH}"
if [ -z "$MODEL" ]; then
  echo "MODEL is not set. Pass as first argument or set MODEL_PATH."
  exit 1
fi

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "NNODES: $NNODES"
echo "GPU: $GPU"
echo "MODEL: $MODEL"

if [ "$RANK" -eq 0 ]; then
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --dist-init-addr "${MASTER_ADDR}:5000" \
        --nnodes "${NNODES}" \
        --node-rank "${RANK}" \
        --host 127.0.0.1 \
        --port 30011 \
        --reasoning-parser qwen3 \
        --tp "${GPU}" \
        --trust-remote-code \
        --max-running-requests 256 \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 16384
else
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --dist-init-addr "${MASTER_ADDR}:5000" \
        --nnodes "${NNODES}" \
        --node-rank "${RANK}" \
        --host 127.0.0.1 \
        --port 30011 \
        --reasoning-parser qwen3 \
        --tp "${GPU}" \
        --trust-remote-code \
        --max-running-requests 256 \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 16384
fi
