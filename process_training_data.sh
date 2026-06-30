#!/usr/bin/env bash
# Usage:
#   bash process_training_data.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DATASETS=("triviaqa_7b" "triviaqa_13b" "triviaqa_70b" "triviaqa_mmlu_13b")
SFT_METHODS=("absolute" "confidence-verb" "confidence-num" "multisample" "sft-baseline")

DATASET=${DATASETS[1]}

# ── 1. Install Python dependencies ────────────────────────────────────────────
echo "==> Installing Python dependencies..."
pip install --quiet torch transformers tqdm

# ── 2. Verify raw training data ───────────────────────────────────────────────
echo ""
echo "==> Verifying training data file..."

TRAIN_FILE="${DATASET}.jsonl"
TRAIN_PATH="data/training_data/$TRAIN_FILE"

if [[ -f "$TRAIN_PATH" ]]; then
    count=$(wc -l < "$TRAIN_PATH")
    echo "  OK  $TRAIN_FILE  ($count records)"
else
    echo "  MISSING  $TRAIN_FILE"
    exit 1
fi

# ── 3. Process training data ──────────────────────────────────────────────────
# prompt_id is determined automatically by process_training_data.py from the sft_method.

for SFT_METHOD in "${SFT_METHODS[@]}"; do
    echo ""
    echo "==> Processing (dataset=$DATASET, method=$SFT_METHOD)..."
    python train/process_training_data.py \
        --dataset_name "$DATASET" \
        --sft_method "$SFT_METHOD" \
        --train_data_path "$TRAIN_PATH" \
        --data_dir "data/processed"
done

echo ""
echo "==> Done."
