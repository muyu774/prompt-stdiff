"""Build dynamic semantic event bank for strict time-truncated semantic conditioning.

Input CSV should contain timestamp and event text/context columns, optionally node_index.
Output NPZ contains:
- step_idx: [E]
- node_index: [E] (-1 for global events)
- embedding: [E, D]
- event_type_id: [E]
- source_id: [E]
- event_type_vocab: [C1]
- source_vocab: [C2]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _clean_text(v: object) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na", "n/a", "unknown"}:
        return None
    return s


def _compose_event_text(row: pd.Series, text_col: Optional[str], fields: List[str]) -> Optional[str]:
    """Build one event text from available columns."""
    if text_col is not None and text_col in row.index:
        text = _clean_text(row[text_col])
        if text is not None:
            return text

    parts: List[str] = []
    for f in fields:
        if f not in row.index:
            continue
        val = _clean_text(row[f])
        if val is not None:
            parts.append(f"{f}: {val}")

    if not parts:
        return None
    return "; ".join(parts)


def _encode_categories(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encode string categories to integer ids with a deterministic vocabulary."""
    vals = values.fillna("unknown").astype(str)
    vocab = sorted(set(vals.tolist()))
    lookup = {v: i for i, v in enumerate(vocab)}
    ids = np.asarray([lookup[v] for v in vals.tolist()], dtype=np.int64)
    vocab_arr = np.asarray(vocab, dtype="<U64")
    return ids, vocab_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic semantic bank")
    parser.add_argument("--events_csv", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-roberta-large-v1")
    parser.add_argument("--time_col", type=str, default="timestamp")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--node_col", type=str, default="node_index")
    parser.add_argument("--event_type_col", type=str, default="event_type")
    parser.add_argument("--source_col", type=str, default="source")
    parser.add_argument(
        "--fields",
        type=str,
        default="weather,incident,holiday,time_context,district,event_type,description,source",
        help="Fallback columns used to compose text when text_col is empty.",
    )
    parser.add_argument("--start_time", type=str, required=True, help="Timeline origin, e.g. 2018-01-01 00:00:00")
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional HuggingFace cache directory for model files.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load the encoder only from local files/cache without network access.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='SentenceTransformer device, e.g. "cpu", "cuda:0", or "auto".',
    )
    parser.add_argument("--normalize_embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Please install sentence-transformers and datasets to build dynamic semantic bank."
        ) from exc

    if not args.events_csv.exists():
        raise FileNotFoundError(
            f"events_csv not found: {args.events_csv}. "
            "Create one first, e.g.\n"
            "python scripts/init_dynamic_events.py "
            "--data_npz data/pems03/data.npz "
            "--out_csv data/pems03/dynamic_events.csv "
            "--mode time_context "
            "--start_time '2018-01-01 00:00:00' --freq_minutes 5"
        )

    df = _normalize_columns(pd.read_csv(args.events_csv))

    time_col = args.time_col.strip().lower()
    text_col = args.text_col.strip().lower() if args.text_col else None
    node_col = args.node_col.strip().lower() if args.node_col else None
    event_type_col = args.event_type_col.strip().lower() if args.event_type_col else None
    source_col = args.source_col.strip().lower() if args.source_col else None
    fields = [x.strip().lower() for x in args.fields.split(",") if x.strip()]

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}. available={list(df.columns)}")

    ts = pd.to_datetime(df[time_col], errors="coerce")
    valid_time = ts.notna()
    df = df.loc[valid_time].copy()
    ts = ts.loc[valid_time]

    start_ts = pd.Timestamp(args.start_time)
    delta_sec = (ts - start_ts).dt.total_seconds().to_numpy()
    step_idx = np.floor(delta_sec / float(args.freq_minutes * 60)).astype(np.int64)

    # ASSUMPTION: events before timeline start are dropped.
    valid_step = step_idx >= 0
    df = df.loc[valid_step].copy()
    step_idx = step_idx[valid_step]

    texts: List[str] = []
    keep_mask = []
    for _, row in df.iterrows():
        text = _compose_event_text(row=row, text_col=text_col, fields=fields)
        if text is None:
            keep_mask.append(False)
            texts.append("")
        else:
            keep_mask.append(True)
            texts.append(text)

    keep_mask_arr = np.asarray(keep_mask, dtype=bool)
    if keep_mask_arr.sum() == 0:
        raise ValueError("No valid event texts found after filtering.")

    df = df.loc[keep_mask_arr].reset_index(drop=True)
    step_idx = step_idx[keep_mask_arr]
    texts = [t for t, k in zip(texts, keep_mask) if k]

    if node_col is not None and node_col in df.columns:
        node_vals = pd.to_numeric(df[node_col], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    else:
        # ASSUMPTION: missing node column means global events.
        node_vals = np.full((len(df),), -1, dtype=np.int64)

    if event_type_col is not None and event_type_col in df.columns:
        event_type_id, event_type_vocab = _encode_categories(df[event_type_col])
    else:
        event_type_id = np.zeros((len(df),), dtype=np.int64)
        event_type_vocab = np.asarray(["unknown"], dtype="<U64")

    if source_col is not None and source_col in df.columns:
        source_id, source_vocab = _encode_categories(df[source_col])
    else:
        source_id = np.zeros((len(df),), dtype=np.int64)
        source_vocab = np.asarray(["unknown"], dtype="<U64")

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_kwargs: dict[str, object] = {}
    if args.cache_dir is not None:
        model_kwargs["cache_folder"] = str(args.cache_dir)
    if args.device != "auto":
        model_kwargs["device"] = args.device
    if args.local_files_only:
        model_kwargs["local_files_only"] = True

    try:
        model = SentenceTransformer(args.model_name, **model_kwargs)
    except TypeError:
        # ASSUMPTION: older sentence-transformers versions may not expose
        # local_files_only in the constructor; offline env vars above still
        # prevent network access in the HuggingFace stack.
        model_kwargs.pop("local_files_only", None)
        model = SentenceTransformer(args.model_name, **model_kwargs)
    except OSError as exc:
        cache_msg = f" with cache_dir={args.cache_dir}" if args.cache_dir is not None else ""
        raise OSError(
            f"Failed to load semantic encoder '{args.model_name}'{cache_msg}. "
            "If the server network is unstable, pass a local model directory via "
            "`--model_name /path/to/all-roberta-large-v1 --local_files_only`."
        ) from exc

    emb = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=int(args.batch_size),
        normalize_embeddings=bool(args.normalize_embeddings),
        show_progress_bar=True,
    ).astype(np.float32)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        step_idx=step_idx.astype(np.int64),
        node_index=node_vals.astype(np.int64),
        embedding=emb,
        event_type_id=event_type_id.astype(np.int64),
        source_id=source_id.astype(np.int64),
        event_type_vocab=event_type_vocab,
        source_vocab=source_vocab,
    )

    print(f"Saved dynamic semantic bank: {args.out_npz}")
    print(f"events={len(step_idx)} dim={emb.shape[1]}")
    print(f"step_idx range=({int(step_idx.min())}, {int(step_idx.max())})")
    print(f"global_event_ratio={(node_vals < 0).mean():.4f}")
    print(f"event_type_vocab={event_type_vocab.tolist()}")
    print(f"source_vocab={source_vocab.tolist()}")


if __name__ == "__main__":
    main()
