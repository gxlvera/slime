NLAYERS=27
FIRST_K_DENSE_REPLACE=1

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# Kimi-VL-A3B-Thinking (text config)
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers 27
    --hidden-size 2048
    --ffn-hidden-size 11264
    --num-attention-heads 16
    --kv-channels 128
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 163840

    --multi-latent-attention
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-base 800000
    --attention-softmax-in-fp32
    --no-rope-fusion

    # moe
    --num-experts 64
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size 1408
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 1408
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0.001
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 2.446
    --moe-router-dtype fp32
    --moe-permute-fusion
)
