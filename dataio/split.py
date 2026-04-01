"""Dataset split helpers."""

from __future__ import annotations

from typing import Tuple


def split_time_series(
    total_steps: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Split a sequence by time into train/val/test ranges.

    Args:
        total_steps: Total time steps.
        train_ratio: Train ratio.
        val_ratio: Validation ratio.
        test_ratio: Test ratio.

    Returns:
        ((train_start, train_end), (val_start, val_end), (test_start, test_end)).
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    train_end = int(total_steps * train_ratio)
    val_end = train_end + int(total_steps * val_ratio)

    return (0, train_end), (train_end, val_end), (val_end, total_steps)
