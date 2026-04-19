#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_log_prob_reject_sweep}"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  "reject_t003:${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_t003.sh"
  "reject_t004:${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_t004.sh"
  "reject_t005:${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_t005.sh"
  "reject_t006:${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_t006.sh"
  "reject_t008:${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_t008.sh"
  "reject_t011:${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_t011.sh"
)

for spec in "${SCRIPTS[@]}"; do
  label="${spec%%:*}"
  script_path="${spec#*:}"
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  run_name="qwen2.5-1.5b-gsm8k-grpo-async-sync-t002-${label}-lr4e-6-${timestamp}"
  log_file="${LOG_DIR}/${run_name}.log"

  echo
  echo "=== Running ${label} ==="
  echo "RUN_NAME=${run_name}"
  echo "LOG_FILE=${log_file}"

  RUN_NAME="${run_name}" WANDB_GROUP="${WANDB_GROUP:-log-prob-reject-threshold-sweep}" \
    bash "${script_path}" "$@" 2>&1 | tee "${log_file}"
done
