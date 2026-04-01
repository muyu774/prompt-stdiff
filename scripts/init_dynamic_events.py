"""Initialize dynamic event CSV for semantic event bank building.

Modes:
- template: create empty/sparse template rows for manual editing.
- time_context: auto-generate global time-context events from traffic timeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _load_total_steps(data_npz: Path) -> int:
    bundle = np.load(data_npz)
    for key in ("data", "x", "arr_0"):
        if key in bundle:
            arr = bundle[key]
            break
    else:
        raise KeyError(f"No supported key in {data_npz}. keys={bundle.files}")

    if arr.ndim < 1:
        raise ValueError(f"Unexpected data shape: {arr.shape}")
    return int(arr.shape[0])


def _time_context_label(ts: pd.Timestamp) -> str:
    hour = ts.hour
    minute = ts.minute
    hm = hour + minute / 60.0

    if 7.0 <= hm <= 9.5:
        return "morning_rush_hour"
    if 17.0 <= hm <= 19.5:
        return "evening_rush_hour"
    if 0.0 <= hm < 5.0:
        return "night_low_demand"
    return "off_peak"


def _event_type_from_row(time_context: str, is_weekend: bool, is_holiday: bool) -> str:
    if is_holiday:
        return "holiday_context"
    if is_weekend:
        return "weekend_context"
    if "rush" in time_context:
        return "rush_hour_context"
    return "weekday_context"


def build_template_rows(start_time: pd.Timestamp, freq_minutes: int, num_rows: int) -> pd.DataFrame:
    rows = []
    for i in range(num_rows):
        ts = start_time + pd.Timedelta(minutes=freq_minutes * i)
        rows.append(
            {
                "timestamp": ts,
                "node_index": -1,
                "text": "",
                "weather": "",
                "incident": "",
                "holiday": "",
                "time_context": "",
                "district": "",
                "event_type": "",
                "description": "",
            }
        )
    return pd.DataFrame(rows)


def build_time_context_rows(
    start_time: pd.Timestamp,
    total_steps: int,
    freq_minutes: int,
    stride_steps: int,
    holiday_dates: List[pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    holiday_days = {d.normalize() for d in holiday_dates}

    for step in range(0, total_steps, max(1, stride_steps)):
        ts = start_time + pd.Timedelta(minutes=freq_minutes * step)
        is_weekend = ts.dayofweek >= 5
        is_holiday = ts.normalize() in holiday_days
        time_context = _time_context_label(ts)

        event_type = _event_type_from_row(
            time_context=time_context,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
        )

        text = (
            f"time_context: {time_context}; "
            f"calendar: {'holiday' if is_holiday else ('weekend' if is_weekend else 'weekday')}"
        )

        rows.append(
            {
                "timestamp": ts,
                "node_index": -1,
                "text": text,
                "weather": "unknown",
                "incident": "none",
                "holiday": "yes" if is_holiday else "no",
                "time_context": time_context,
                "district": "global",
                "event_type": event_type,
                "description": "auto-generated temporal context event",
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize dynamic events CSV")
    parser.add_argument("--data_npz", type=Path, required=True, help="Traffic data npz path")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output events CSV path")
    parser.add_argument("--mode", type=str, default="time_context", choices=["template", "time_context"])
    parser.add_argument("--start_time", type=str, required=True, help="Timeline start time, e.g. 2018-01-01 00:00:00")
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument("--stride_steps", type=int, default=1, help="Only for time_context mode")
    parser.add_argument(
        "--holiday_dates",
        type=str,
        default="",
        help="Comma-separated holiday dates (YYYY-MM-DD), e.g. 2018-01-01,2018-02-16",
    )
    parser.add_argument("--template_rows", type=int, default=128, help="Only for template mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_time = pd.Timestamp(args.start_time)
    total_steps = _load_total_steps(args.data_npz)

    holiday_dates: List[pd.Timestamp] = []
    if args.holiday_dates.strip():
        holiday_dates = [pd.Timestamp(x.strip()) for x in args.holiday_dates.split(",") if x.strip()]

    if args.mode == "template":
        df = build_template_rows(
            start_time=start_time,
            freq_minutes=int(args.freq_minutes),
            num_rows=int(args.template_rows),
        )
    else:
        df = build_time_context_rows(
            start_time=start_time,
            total_steps=total_steps,
            freq_minutes=int(args.freq_minutes),
            stride_steps=int(args.stride_steps),
            holiday_dates=holiday_dates,
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved dynamic events CSV: {args.out_csv}")
    print(f"rows={len(df)} mode={args.mode}")


if __name__ == "__main__":
    main()
