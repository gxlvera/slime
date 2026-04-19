#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LOG_PROB_REJECT_THRESHOLD="${LOG_PROB_REJECT_THRESHOLD:-0.05}"
exec bash "${SCRIPT_DIR}/run_qwen2_5_1_5b_gsm8k_grpo_async_log_prob_sync_t002_reject_common.sh" "$@"
