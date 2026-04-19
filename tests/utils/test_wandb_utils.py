import pytest

from slime.utils.wandb_utils import WANDB_GROUP_MAX_LENGTH, _truncate_wandb_group


@pytest.mark.unit
def test_truncate_wandb_group_keeps_short_group_unchanged():
    group = "qwen2.5-1.5b-gsm8k-grpo-async"

    assert _truncate_wandb_group(group) == group


@pytest.mark.unit
def test_truncate_wandb_group_caps_long_group_with_stable_hash():
    group = "gxl/" + "long-ablation-name-" * 10

    truncated = _truncate_wandb_group(group)

    assert len(truncated) == WANDB_GROUP_MAX_LENGTH
    assert truncated == _truncate_wandb_group(group)
    assert truncated.startswith("gxl/long-ablation-name-")
    assert truncated != group[:WANDB_GROUP_MAX_LENGTH]
