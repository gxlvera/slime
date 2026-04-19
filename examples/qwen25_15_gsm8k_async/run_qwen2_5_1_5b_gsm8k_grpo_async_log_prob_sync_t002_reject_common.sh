#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

LOG_PROB_SYNC_THRESHOLD="${LOG_PROB_SYNC_THRESHOLD:-0.02}"
: "${LOG_PROB_REJECT_THRESHOLD:?LOG_PROB_REJECT_THRESHOLD is not set.}"
LR="${LR:-4e-6}"

SETTING="sync-t${LOG_PROB_SYNC_THRESHOLD}_reject-t${LOG_PROB_REJECT_THRESHOLD}_lr-${LR}"

export WANDB_GROUP="${WANDB_GROUP:-log-prob-reject-threshold-sweep}"
export RUN_NAME="${RUN_NAME:-qwen2.5-1.5b-gsm8k-grpo-async-${SETTING}}"

exec bash "${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_staleness_uwi_phase1_common.sh" \
  --lr "${LR}" \
  --update-weights-logprob-diff-threshold "${LOG_PROB_SYNC_THRESHOLD}" \
  --update-weights-logprob-diff-reject-threshold "${LOG_PROB_REJECT_THRESHOLD}" \
  "$@"
