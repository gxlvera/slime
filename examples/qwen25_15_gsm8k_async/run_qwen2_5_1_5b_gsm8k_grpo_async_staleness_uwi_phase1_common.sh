#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_NAME="${RUN_NAME:-qwen2.5-1.5b-gsm8k-grpo-async-staleness-uwi-phase1}"
WANDB_TEAM="${WANDB_TEAM:-fenglin02}"
WANDB_PROJECT="${WANDB_PROJECT:-15642-final-smas4ar}"
WANDB_GROUP="${WANDB_GROUP:?WANDB_GROUP is not set. Please set it in the calling script.}"
NUM_ROLLOUT="${NUM_ROLLOUT:-160}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-40}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async.py" \
  --mode async_train \
  --run-name "${RUN_NAME}" \
  --wandb-team "${WANDB_TEAM}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --rollout-max-response-len 512 \
  --eval-max-response-len 768 \
  --max-tokens-per-gpu 3072 \
  --eval-interval "${EVAL_INTERVAL}" \
  --lr 3e-6 \
  --dynamic-sampling-filter \
  "$@"
  # --get-mismatch-metrics \
  # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp \
