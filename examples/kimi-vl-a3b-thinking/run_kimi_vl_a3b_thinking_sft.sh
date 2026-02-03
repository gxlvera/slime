TRAIN_BACKEND="megatron"
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}


# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


# Common args
CKPT_ARGS=(
   --hf-checkpoint /root/Kimi-VL-A3B-Thinking-2506
   --load /root/Kimi-VL-A3B-Thinking-2506
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/datasets/geo3k_imgurl/train_formatted.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 3000
   --rollout-batch-size 128
   --global-batch-size 128
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'


OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-dev
        --wandb-group kimi-vl-a3b-thinking-sft
        --wandb-key ${WANDB_API_KEY}
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

# Backend-specific args
if [ "$TRAIN_BACKEND" = "fsdp" ]; then
    BACKEND_ARGS=(
      --train-backend fsdp
      --gradient-checkpointing
      --attn-implementation flash_attention_3
      --update-weight-buffer-size 536870912
    )
else
    # megatron backend (default)
    BACKEND_ARGS=(
      --train-backend megatron
      --tensor-model-parallel-size 1
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 1
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu 4096
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash
      --megatron-to-hf-mode bridge
    )

   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
   source "${SCRIPT_DIR}/../../scripts/models/moonlight.sh"
   FILTERED_MODEL_ARGS=()
   for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
      if [ "${MODEL_ARGS[$i]}" = "--rotary-base" ]; then
         i=$((i+1))
         continue
      fi
      FILTERED_MODEL_ARGS+=("${MODEL_ARGS[$i]}")
   done
   MODEL_ARGS=("${FILTERED_MODEL_ARGS[@]}" --rotary-base 800000)
fi

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-Bridge/src:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]}
