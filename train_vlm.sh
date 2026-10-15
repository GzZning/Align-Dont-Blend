#!/usr/bin/env bash
set -euo pipefail

: "${VLM_MODEL:?Set VLM_MODEL to the base-model directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for training checkpoints}"
SEMANTIC_DATASET="${SEMANTIC_DATASET:-data/semantic_preference_train.jsonl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
VLM_DTYPE="${VLM_DTYPE:-float16}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
GLOBAL_BATCH_SIZE=8

[[ -f "$SEMANTIC_DATASET" ]] || { echo "Missing dataset: $SEMANTIC_DATASET" >&2; exit 1; }
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid NPROC_PER_NODE" >&2; exit 1; }
(( GLOBAL_BATCH_SIZE % NPROC_PER_NODE == 0 )) || { echo "Use 1, 2, 4, or 8 workers" >&2; exit 1; }
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))
export NPROC_PER_NODE
export MAX_PIXELS

swift rlhf \
  --rlhf_type dpo \
  --model "$VLM_MODEL" \
  --dataset "$SEMANTIC_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "$VLM_DTYPE" \
  --max_pixels "$MAX_PIXELS" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --deepspeed zero3 \
  --max_length 8192 \
  --beta 0.05 \
  --loss_type sigmoid \
  --rpo_alpha 0
