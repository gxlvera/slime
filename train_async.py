import logging
import math
import time

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, log, update_tracking_open_metrics
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _extract_actor_logprob_diff(train_results):
    for result in train_results:
        if isinstance(result, dict) and "train/train_rollout_logprob_abs_diff" in result:
            return result["train/train_rollout_logprob_abs_diff"]
    return None


def _extract_actor_rejected_by_logprob_diff(train_results):
    for result in train_results:
        if isinstance(result, dict) and result.get("train/rollout_rejected_by_logprob_diff", 0):
            return True
    return False


def _as_integral_interval(value, name):
    rounded = round(value)
    if rounded < 1 or not math.isclose(value, rounded):
        raise ValueError(f"{name} must resolve to a positive integer, got {value}.")
    return rounded


def _get_update_weights_interval(args, rollout_id):
    if args.update_weights_interval_decay_rate is None:
        return args.update_weights_interval

    decay_rate = args.update_weights_interval_decay_rate
    rollout_step = rollout_id + 1
    in_base_interval_phase = rollout_step <= args.num_rollout * 2 / 3
    if in_base_interval_phase:
        return args.update_weights_interval
    return _as_integral_interval(
        args.update_weights_interval * decay_rate,
        "update_weights_interval * update_weights_interval_decay_rate",
    )


def _get_update_weights_logprob_diff_threshold(args, rollout_id):
    if args.update_weights_logprob_diff_threshold_decay_to is None:
        return args.update_weights_logprob_diff_threshold

    rollout_step = rollout_id + 1
    in_base_threshold_phase = rollout_step <= args.num_rollout * args.update_weights_logprob_diff_threshold_decay_after_ratio
    if in_base_threshold_phase:
        return args.update_weights_logprob_diff_threshold
    return args.update_weights_logprob_diff_threshold_decay_to


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    initial_update_weights_count = 0
    training_update_weights_count = 0

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()
        initial_update_weights_count += 1

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    train_start_time = time.monotonic()
    eval_time = 0.0

    def eval_rollout(rollout_id):
        nonlocal eval_time
        eval_start_time = time.monotonic()
        try:
            return ray.get(rollout_manager.eval.remote(rollout_id))
        finally:
            eval_time += time.monotonic() - eval_start_time

    # async train loop. `rollout_id` is the logical training step and only
    # advances after a batch is actually used for optimization. Generation ids
    # are physical attempts and can advance further when batches are rejected.
    next_generation_rollout_id = args.start_rollout_id

    def start_generation():
        nonlocal next_generation_rollout_id
        future = rollout_manager.generate.remote(next_generation_rollout_id)
        next_generation_rollout_id += 1
        return future

    rollout_data_curr_ref = None
    rollout_data_curr_future = start_generation()
    suppress_next_logprob_diff_sync = False
    rollout_id = args.start_rollout_id
    staleness_attempt_step = 0
    while rollout_id < args.num_rollout:
        actor_train_logprob_diff = None
        actor_batch_rejected = False
        suppress_current_staleness_action = suppress_next_logprob_diff_sync
        suppress_next_logprob_diff_sync = False

        if rollout_data_curr_ref is None:
            rollout_data_curr_ref = ray.get(rollout_data_curr_future)
            rollout_data_curr_future = None

        # Start the next rollout early.
        rollout_data_next_future = None
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = start_generation()

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
                actor_train_logprob_diff = _extract_actor_logprob_diff(actor_train_results)
                actor_batch_rejected = _extract_actor_rejected_by_logprob_diff(actor_train_results)
            ray.get(critic_train_handle)
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            actor_train_logprob_diff = _extract_actor_logprob_diff(actor_train_results)
            actor_batch_rejected = _extract_actor_rejected_by_logprob_diff(actor_train_results)

        logprob_diff_threshold = None
        if args.update_weights_logprob_diff_threshold is not None:
            logprob_diff_threshold = _get_update_weights_logprob_diff_threshold(args, rollout_id)

        if actor_train_logprob_diff is not None:
            staleness_attempt_step += 1
            rollout_step = compute_rollout_step(args, rollout_id)
            log(
                args,
                {
                    "staleness/attempt_step": staleness_attempt_step,
                    "staleness/logprob_diff_actual": actor_train_logprob_diff,
                    "staleness/logprob_diff_sync_threshold": logprob_diff_threshold,
                    "staleness/logprob_diff_reject_threshold": args.update_weights_logprob_diff_reject_threshold,
                    "staleness/batch_rejected": int(actor_batch_rejected),
                    "staleness/batch_used_for_training": int(not actor_batch_rejected),
                    "staleness/suppress_batch": int(suppress_current_staleness_action),
                    "staleness/cascade_discard_next": int(
                        actor_batch_rejected and not suppress_current_staleness_action
                    ),
                    "staleness/logical_rollout_step": rollout_step,
                },
                step_key="staleness/attempt_step",
            )

        if actor_batch_rejected:
            logger.info(
                "Reject rollout batch by logprob diff at rollout_id %s: diff=%s, suppress_current=%s",
                rollout_id,
                actor_train_logprob_diff,
                suppress_current_staleness_action,
            )
            if suppress_current_staleness_action:
                # The current batch was prefetched before a just-finished weight sync.
                # Drop only this stale batch; keep the already-prefetched next batch,
                # which was generated after the sync.
                rollout_data_curr_ref = None
                rollout_data_curr_future = rollout_data_next_future
                if rollout_data_curr_future is None:
                    rollout_data_curr_future = start_generation()
            else:
                # Current and next were both generated from stale rollout weights.
                # Drain the next generation to avoid syncing weights mid-generation,
                # discard it, then sync and retry the same logical rollout id.
                if rollout_data_next_future is not None:
                    ray.get(rollout_data_next_future)
                if not args.critic_train_only:
                    actor_model.update_weights()
                    training_update_weights_count += 1
                rollout_data_curr_ref = None
                rollout_data_curr_future = start_generation()
            continue

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if not args.critic_train_only:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        update_weights_interval = args.update_weights_interval
        interval_hit = False
        if args.update_weights_logprob_diff_threshold is None:
            update_weights_interval = _get_update_weights_interval(args, rollout_id)
            interval_hit = (rollout_id + 1) % update_weights_interval == 0
        diff_hit = (
            args.update_weights_logprob_diff_threshold is not None
            and actor_train_logprob_diff is not None
            and actor_train_logprob_diff > logprob_diff_threshold
        )
        logprob_diff_sync_suppressed = False
        if args.update_weights_logprob_diff_threshold is not None:
            logprob_diff_sync_suppressed = diff_hit and suppress_current_staleness_action
            should_update_weights = diff_hit and not logprob_diff_sync_suppressed
        else:
            should_update_weights = interval_hit

        if args.update_weights_logprob_diff_threshold is not None:
            rollout_step = compute_rollout_step(args, rollout_id)
            log(
                args,
                {
                    "rollout/step": rollout_step,
                    "rollout/update_weights_by_logprob_diff_enabled": 1,
                    "rollout/update_weights_by_logprob_diff_threshold": logprob_diff_threshold,
                    "rollout/update_weights_by_logprob_diff_base_threshold": args.update_weights_logprob_diff_threshold,
                    "rollout/update_weights_by_logprob_diff_decay_to": (
                        args.update_weights_logprob_diff_threshold_decay_to
                    ),
                    "rollout/update_weights_by_logprob_diff_decay_after_ratio": (
                        args.update_weights_logprob_diff_threshold_decay_after_ratio
                    ),
                    "rollout/update_weights_by_logprob_diff_value": actor_train_logprob_diff,
                    "rollout/update_weights_by_logprob_diff_triggered": int(should_update_weights),
                    "rollout/update_weights_by_logprob_diff_suppressed": int(logprob_diff_sync_suppressed),
                    "rollout/logprob_diff_used": actor_train_logprob_diff,
                    "rollout/logprob_diff_used_sync_threshold": logprob_diff_threshold,
                    "rollout/logprob_diff_used_reject_threshold": args.update_weights_logprob_diff_reject_threshold,
                },
                step_key="rollout/step",
            )
        elif args.update_weights_interval_decay_rate is not None:
            rollout_step = compute_rollout_step(args, rollout_id)
            log(
                args,
                {
                    "rollout/step": rollout_step,
                    "rollout/update_weights_by_decay_interval_enabled": 1,
                    "rollout/update_weights_by_decay_interval_base": args.update_weights_interval,
                    "rollout/update_weights_by_decay_interval_rate": args.update_weights_interval_decay_rate,
                    "rollout/update_weights_by_decay_interval_current": update_weights_interval,
                    "rollout/update_weights_by_decay_interval_triggered": int(should_update_weights),
                },
                step_key="rollout/step",
            )

        if should_update_weights:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(rollout_data_next_future) if rollout_data_next_future is not None else None
            rollout_data_curr_future = None
            if not args.critic_train_only:
                actor_model.update_weights()
                training_update_weights_count += 1
                if args.update_weights_logprob_diff_threshold is not None:
                    suppress_next_logprob_diff_sync = True
        else:
            rollout_data_curr_ref = None
            rollout_data_curr_future = rollout_data_next_future

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            eval_rollout(rollout_id)

        rollout_id += 1

    if args.num_rollout > args.start_rollout_id:
        final_rollout_id = args.num_rollout - 1
    else:
        final_rollout_id = args.start_rollout_id
    rollout_step = compute_rollout_step(args, final_rollout_id)
    total_training_time = max(time.monotonic() - train_start_time - eval_time, 0.0)
    log_dict = {
        "rollout/step": rollout_step,
        "rollout/update_weights_total": training_update_weights_count,
        "rollout/update_weights_total_including_initial": (
            initial_update_weights_count + training_update_weights_count
        ),
        "perf/total_training_time": total_training_time,
    }
    logger.info(f"perf total training: {log_dict}")
    log(
        args,
        log_dict,
        step_key="rollout/step",
    )

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
