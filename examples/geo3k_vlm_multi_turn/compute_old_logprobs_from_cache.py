import json
import os
import sys
from datetime import timedelta
from pathlib import Path

MEGATRON_PATH = os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
sys.path.insert(0, MEGATRON_PATH)
sys.path.insert(0, "/root/slime")

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from slime.utils.arguments import parse_args
from slime.backends.megatron_utils.initialize import init as megatron_init
from slime.backends.megatron_utils.model import initialize_model_and_optimizer, forward_only
from slime.backends.megatron_utils.data import get_data_iterator
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.distributed_utils import init_gloo_group


def _build_args(model_path: str, batch_size: int, num_batches: int):
    args_list = [
        "--train-backend", "megatron",
        "--swiglu",
        "--num-layers", "28",
        "--hidden-size", "2048",
        "--ffn-hidden-size", "6144",
        "--num-attention-heads", "16",
        "--group-query-attention",
        "--num-query-groups", "8",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", os.environ.get("MODEL_ARGS_ROTARY_BASE", "1000000"),
        "--vocab-size", "151936",
        "--kv-channels", "128",
        "--qk-layernorm",
        "--hf-checkpoint", model_path,
        "--load", model_path,
        "--actor-num-gpus-per-node", "1",
        "--num-gpus-per-node", "1",
        "--rollout-batch-size", str(batch_size),
        "--num-rollout", str(num_batches),
        "--global-batch-size", str(batch_size),
        "--micro-batch-size", str(batch_size),
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        "--expert-tensor-parallel-size", "1",
        "--attention-backend", "flash",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--megatron-to-hf-mode", "bridge",
    ]

    argv_backup = sys.argv
    try:
        sys.argv = [argv_backup[0]] + args_list
        args = parse_args()
    finally:
        sys.argv = argv_backup

    if getattr(args, "load", None) is None:
        args.load = model_path

    return args


def _load_cache(cache_path: Path, max_samples: int):
    rows = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("index", 0))
    return rows[:max_samples]


def _to_cuda_mm(mm_dict):
    if mm_dict is None:
        return None
    return {k: v.to(device=torch.cuda.current_device()) for k, v in mm_dict.items()}


def main():
    model_path = os.environ.get("MODEL_PATH", "/root/models/Qwen3-VL-2B-Instruct")
    cache_path = Path(
        os.environ.get(
            "ROLLOUT_CACHE_PATH",
            "/root/slime/examples/geo3k_vlm_multi_turn/batch_compare_outputs/sglang_rollout_logged.jsonl",
        )
    )
    mode = os.environ.get("MODE")
    if not mode:
        raise RuntimeError("MODE env var is required (used as output filename).")
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    num_batches = int(os.environ.get("NUM_BATCHES", "1"))

    output_dir = Path(
        "/root/slime/examples/geo3k_vlm_multi_turn/batch_compare_outputs/log_prob_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / f"{mode}.jsonl"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    torch.cuda.set_device("cuda:0")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    init_gloo_group()

    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"

    args = _build_args(model_path, batch_size, num_batches)
    args.rank = 0
    args.world_size = 1
    megatron_init(args)

    model, _, _, _ = initialize_model_and_optimizer(args, role="actor")

    samples = _load_cache(cache_path, batch_size * num_batches)
    if len(samples) < batch_size * num_batches:
        raise RuntimeError(f"Cache has only {len(samples)} samples, expected {batch_size * num_batches}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if output_jsonl.exists():
        output_jsonl.unlink()

    for batch_id in range(num_batches):
        batch_samples = samples[batch_id * batch_size : (batch_id + 1) * batch_size]

        tokens_list = []
        loss_masks = []
        response_lengths = []
        total_lengths = []
        mm_inputs = []
        sample_indices = []

        for s in batch_samples:
            tokens = s["tokens"]
            loss_mask = s["loss_mask"]
            response_length = s["response_length"]
            total_length = len(tokens)
            mm_path = s.get("multimodal_train_inputs_path")
            mm_dict = torch.load(mm_path, map_location="cpu") if mm_path else None

            tokens_list.append(torch.tensor(tokens, dtype=torch.long, device=torch.cuda.current_device()))
            loss_masks.append(torch.tensor(loss_mask, dtype=torch.int, device=torch.cuda.current_device()))
            response_lengths.append(response_length)
            total_lengths.append(total_length)
            mm_inputs.append(_to_cuda_mm(mm_dict))
            sample_indices.append(s.get("index"))

        rollout_data = {
            "tokens": tokens_list,
            "loss_masks": loss_masks,
            "response_lengths": response_lengths,
            "total_lengths": total_lengths,
            "multimodal_train_inputs": mm_inputs,
            "sample_indices": sample_indices,
        }

        data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
        outputs = forward_only(
            get_log_probs_and_entropy,
            args,
            model,
            data_iterator,
            num_microbatches,
        )

        log_probs_list = outputs["log_probs"]
        with output_jsonl.open("a", encoding="utf-8") as f:
            for pos, (s, lp, lm) in enumerate(zip(batch_samples, log_probs_list, loss_masks, strict=False)):
                lm_sum = torch.clamp_min(lm.sum(), 1)
                avg_lp = (lp * lm).sum() / lm_sum

                response_tokens = s["tokens"][-s["response_length"] :]
                f.write(
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "sample_pos": pos,
                            "sample_index": s.get("index"),
                            "response_length": s["response_length"],
                            "total_length": len(s["tokens"]),
                            "old_log_probs": lp.detach().float().cpu().tolist(),
                            "loss_mask": lm.detach().float().cpu().tolist(),
                            "masked_avg_log_prob": float(avg_lp.detach().cpu()),
                            "response_token_ids": response_tokens,
                            "response_tokens": [tokenizer.decode([tid], skip_special_tokens=False) for tid in response_tokens],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"Wrote {output_jsonl}")


if __name__ == "__main__":
    main()
