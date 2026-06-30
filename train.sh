#!/usr/bin/env bash
# Usage:
#   bash train/train.sh [NUM_GPUS]

set -euo pipefail

NUM_GPUS=${1:-4}   # [FLAG] paper does not state GPU count; set to your cluster size

DS_BUILD_CPU_ADAM=1 pip install -r requirements.txt

export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE="${NUM_GPUS}"
export RANK="0"
export LOCAL_RANK="0"

deepspeed \
  --num_gpus "${NUM_GPUS}" \
  train/train.py \
    --train_epochs 2 \
    --total_batch_size 8 \
    --ds_config train/ds_config.json
