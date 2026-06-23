"""Experiment result writer for reproducible benchmark reporting."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


@dataclass
class ExperimentResult:
    """One row of benchmark results.

    Attributes:
        dataset: Dataset name, e.g. pems04 or pems08.
        method: Model or baseline name.
        setting: Run setting, e.g. original, +semantic, full.
        horizon: Forecast horizon in 5-minute steps.
        mae: Mean absolute error.
        rmse: Root mean square error.
        crps: Continuous ranked probability score; blank/None for deterministic baselines.
        seed: Random seed.
        config: Config path or config identifier.
        implementation: official, reimplemented, control, or ours.
        checkpoint: Optional checkpoint path.
        settings_json: Exact run settings serialized as JSON.
        notes: Optional free-form note.
    """

    dataset: str
    method: str
    setting: str
    horizon: int
    mae: float
    rmse: float
    crps: Optional[float]
    seed: int
    config: str
    implementation: str = ""
    checkpoint: str = ""
    settings_json: str = ""
    notes: str = ""


FIELDNAMES: Sequence[str] = (
    "timestamp_utc",
    "dataset",
    "method",
    "setting",
    "horizon",
    "mae",
    "rmse",
    "crps",
    "seed",
    "config",
    "implementation",
    "checkpoint",
    "settings_json",
    "notes",
)


def _fmt_metric(value: Optional[float]) -> str:
    """Format metrics for human-readable markdown tables."""
    if value is None:
        return "--"
    return f"{float(value):.6f}"


def _row_to_csv_dict(result: ExperimentResult, timestamp_utc: str) -> Mapping[str, str]:
    """Convert one result row to CSV-safe strings."""
    data = asdict(result)
    data["timestamp_utc"] = timestamp_utc
    data["mae"] = _fmt_metric(result.mae)
    data["rmse"] = _fmt_metric(result.rmse)
    data["crps"] = "" if result.crps is None else _fmt_metric(result.crps)
    return {key: str(data.get(key, "")) for key in FIELDNAMES}


def results_to_markdown(
    results: Iterable[ExperimentResult],
    title: str = "Experiment Results",
) -> str:
    """Render results as a markdown table."""
    rows = list(results)
    lines: List[str] = [
        f"### {title}",
        "",
        "| Dataset | Method | Setting | Horizon | MAE | RMSE | CRPS | Seed | Implementation | Config |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| "
            f"{r.dataset} | {r.method} | {r.setting} | {int(r.horizon)} | "
            f"{_fmt_metric(r.mae)} | {_fmt_metric(r.rmse)} | {_fmt_metric(r.crps)} | "
            f"{int(r.seed)} | {r.implementation} | `{r.config}` |"
        )
    lines.append("")
    return "\n".join(lines)


def append_results_csv(
    csv_path: Path,
    results: Iterable[ExperimentResult],
    timestamp_utc: Optional[str] = None,
) -> None:
    """Append result rows to a CSV file, creating a header when needed."""
    rows = list(results)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp_utc or datetime.now(timezone.utc).isoformat()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDNAMES))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv_dict(row, timestamp_utc=timestamp))


def append_results_markdown(
    md_path: Path,
    results: Iterable[ExperimentResult],
    title: str = "Experiment Results",
) -> None:
    """Append a markdown results table to RESULTS.md."""
    rows = list(results)
    if not rows:
        return
    md_path.parent.mkdir(parents=True, exist_ok=True)
    table = results_to_markdown(rows, title=title)
    prefix = ""
    if md_path.exists() and md_path.stat().st_size > 0:
        prefix = "\n"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(prefix + table + "\n")


def write_experiment_results(
    results: Iterable[ExperimentResult],
    csv_path: Path = Path("outputs/results.csv"),
    md_path: Path = Path("RESULTS.md"),
    title: str = "Experiment Results",
) -> None:
    """Write results to both CSV and markdown artifacts."""
    rows = list(results)
    if not rows:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    append_results_csv(csv_path=csv_path, results=rows, timestamp_utc=timestamp)
    append_results_markdown(md_path=md_path, results=rows, title=title)
