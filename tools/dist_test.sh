#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [TEST_OPTIONS...]" >&2
    exit 2
fi
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

torchrun \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${PORT:-29501}" \
    --nproc_per_node="$GPUS" \
    "$SCRIPT_DIR/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
