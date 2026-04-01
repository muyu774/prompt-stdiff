"""Window slicing utilities for spatio-temporal traffic data."""

from __future__ import annotations

from typing import List, Tuple


def build_window_indices(
    total_steps: int,
    history_steps: int,
    horizon_steps: int,
) -> List[Tuple[int, int, int]]:
    """Build (his_start, his_end, fut_end) indices for sliding windows.

    Args:
        total_steps: Total sequence length T_total.
        history_steps: Number of history steps T.
        horizon_steps: Number of future steps H.

    Returns:
        A list of tuples with inclusive-exclusive style:
        - history slice: [his_start:his_end]
        - future slice: [his_end:fut_end]
    """
    indices: List[Tuple[int, int, int]] = []
    max_start = total_steps - history_steps - horizon_steps
    for start in range(max_start + 1):
        his_end = start + history_steps
        fut_end = his_end + horizon_steps
        indices.append((start, his_end, fut_end))
    return indices
