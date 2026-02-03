source "/root/slime/scripts/models/moonlight.sh"

PYTHONPATH=/root/Megatron-Bridge/src:/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Kimi-VL-A3B-Thinking-2506 \
  --save /root/Kimi-VL-A3B-Thinking-2506_torch_dist \
  --megatron-to-hf-mode bridge
