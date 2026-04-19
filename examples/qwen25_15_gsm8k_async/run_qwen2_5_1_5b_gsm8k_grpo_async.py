import argparse
import os
from pathlib import Path

import slime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen2.5-1.5B-Instruct"
MODEL_TYPE = "qwen2.5-1.5B"
DATASET_NAME = "zhuzilin/gsm8k"

MODEL_ROOT = Path("/root/models") / MODEL_NAME
DATASET_ROOT = Path("/root/datasets/gsm8k")
TRAIN_PATH = DATASET_ROOT / "train.parquet"
EVAL_PATH = DATASET_ROOT / "test.parquet"


def _maybe_slice(path: Path, slice_expr: str | None) -> str:
    return f"{path}@{slice_expr}" if slice_expr else str(path)


def _wandb_args(run_name: str, team: str | None, project: str, group: str) -> str:
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY is required for this experiment.")
    full_group = f"{group}-{run_name}" if run_name else group
    team_args = f"--wandb-team {team} " if team else ""
    return (
        "--use-wandb "
        f"{team_args}"
        f"--wandb-project {project} "
        f"--wandb-group {full_group} "
        f"--wandb-key '{wandb_api_key}' "
        "--disable-wandb-random-suffix "
    )


def prepare() -> None:
    U.exec_command("mkdir -p /root/models /root/datasets")
    if not MODEL_ROOT.exists():
        U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_ROOT}")
    if not TRAIN_PATH.exists() or not EVAL_PATH.exists():
        U.hf_download_dataset(DATASET_NAME)


def build_train_args(args: argparse.Namespace) -> str:
    ckpt_args = f"--hf-checkpoint {MODEL_ROOT} "
    if args.use_kl_loss:
        ckpt_args += f"--ref-load {MODEL_ROOT} "

    rollout_args = (
        f"--prompt-data {_maybe_slice(TRAIN_PATH, args.train_slice)} "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--global-batch-size {args.global_batch_size} "
    )

    if args.dynamic_sampling_filter:
        rollout_args += (
            "--dynamic-sampling-filter-path "
            "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )
    if args.num_steps_per_rollout is not None:
        rollout_args += f"--num-steps-per-rollout {args.num_steps_per_rollout} "
    if args.get_mismatch_metrics:
        rollout_args += "--get-mismatch-metrics "
        rollout_args += f"--custom-tis-function-path {args.custom_tis_function_path} "

    eval_args = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-prompt-data gsm8k {_maybe_slice(EVAL_PATH, args.eval_slice)} "
        "--n-samples-per-eval-prompt 1 "
        f"--eval-max-response-len {args.eval_max_response_len} "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        f"{'--use-kl-loss ' if args.use_kl_loss else ''}"
        f"--kl-loss-coef {args.kl_loss_coef:.6g} "
        "--kl-loss-type low_var_kl "
        f"--entropy-coef {args.entropy_coef:.6g} "
        f"--eps-clip {args.eps_clip:.6g} "
        f"--eps-clip-high {args.eps_clip_high:.6g} "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr:.6g} "
        "--lr-decay-style constant "
        f"--weight-decay {args.weight_decay:.6g} "
        f"--adam-beta1 {args.adam_beta1:.6g} "
        f"--adam-beta2 {args.adam_beta2:.6g} "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static:.6g} "
        f"--sglang-cuda-graph-max-bs {args.sglang_cuda_graph_max_bs} "
        "--sglang-enable-metrics "
    )

    misc_args = (
        "--rotary-base 1000000 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--update-weights-interval {args.update_weights_interval} "
        "--megatron-to-hf-mode bridge "
    )
    if args.mode == "colocate_train":
        misc_args += "--colocate "

    if args.update_weights_logprob_diff_threshold is not None:
        misc_args += f"--update-weights-logprob-diff-threshold {args.update_weights_logprob_diff_threshold} "
    if args.update_weights_logprob_diff_reject_threshold is not None:
        misc_args += (
            f"--update-weights-logprob-diff-reject-threshold "
            f"{args.update_weights_logprob_diff_reject_threshold} "
        )
    if args.update_weights_logprob_diff_threshold_decay_to is not None:
        misc_args += (
            f"--update-weights-logprob-diff-threshold-decay-to "
            f"{args.update_weights_logprob_diff_threshold_decay_to} "
        )
        misc_args += (
            f"--update-weights-logprob-diff-threshold-decay-after-ratio "
            f"{args.update_weights_logprob_diff_threshold_decay_after_ratio} "
        )
    if args.update_weights_interval_decay_rate is not None:
        misc_args += f"--update-weights-interval-decay-rate {args.update_weights_interval_decay_rate} "

    wandb_args = _wandb_args(args.run_name, args.wandb_team, args.wandb_project, args.wandb_group)

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{wandb_args} "
    )


def run_eval_only(args: argparse.Namespace) -> None:
    train_args = build_train_args(args).replace(f"--num-rollout {args.num_rollout} ", "--num-rollout 0 ")
    train_args += "--lr-decay-iters 1 "
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.actor_num_gpus_per_node + args.rollout_num_gpus,
        megatron_model_type=MODEL_TYPE,
        train_script="train.py",
        extra_env_vars={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
    )


def run_async_train(args: argparse.Namespace) -> None:
    train_args = build_train_args(args)
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.actor_num_gpus_per_node + args.rollout_num_gpus,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        extra_env_vars={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
    )


def run_colocate_train(args: argparse.Namespace) -> None:
    train_args = build_train_args(args)
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
        train_script="train.py",
        extra_env_vars={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-1.5B-Instruct GRPO on GSM8K with async train/infer split.")
    parser.add_argument("--mode", choices=("eval_only", "async_train", "colocate_train"), default="async_train")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-team", default=None)
    parser.add_argument("--wandb-project", default="slime-gsm8k-qwen25")
    parser.add_argument("--wandb-group", default="qwen2.5-1.5b-gsm8k-grpo-async")
    parser.add_argument("--actor-num-gpus-per-node", type=int, default=1)
    parser.add_argument("--rollout-num-gpus", type=int, default=1)
    parser.add_argument("--num-rollout", type=int, default=24)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--rollout-max-response-len", type=int, default=512)
    parser.add_argument("--eval-max-response-len", type=int, default=768)
    parser.add_argument("--rollout-temperature", type=float, default=0.8)
    parser.add_argument("--eval-interval", type=int, default=24)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=3072)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--sglang-cuda-graph-max-bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--kl-loss-coef", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--eps-clip-high", type=float, default=0.28)
    parser.add_argument("--use-kl-loss", action="store_true")
    parser.add_argument("--dynamic-sampling-filter", action="store_true")
    parser.add_argument("--num-steps-per-rollout", type=int, default=None)
    parser.add_argument("--update-weights-interval", type=int, default=1)
    parser.add_argument("--update-weights-interval-decay-rate", type=float, default=None)
    parser.add_argument("--update-weights-logprob-diff-threshold", type=float, default=None)
    parser.add_argument("--update-weights-logprob-diff-reject-threshold", type=float, default=None)
    parser.add_argument("--update-weights-logprob-diff-threshold-decay-to", type=float, default=None)
    parser.add_argument("--update-weights-logprob-diff-threshold-decay-after-ratio", type=float, default=2 / 3)
    parser.add_argument("--get-mismatch-metrics", action="store_true")
    parser.add_argument(
        "--custom-tis-function-path",
        default="examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp",
    )
    parser.add_argument("--train-slice", default=None)
    parser.add_argument("--eval-slice", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare()
    if args.prepare_only:
        return
    if args.mode == "eval_only":
        run_eval_only(args)
    elif args.mode == "colocate_train":
        run_colocate_train(args)
    else:
        run_async_train(args)


if __name__ == "__main__":
    main()
