"""Offline semantic embedding encoder for traffic node prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from semantic.encoder_interface import SemanticEncoder
from semantic.prompt_builder import build_prompts
from semantic.semantic_cache import save_semantic_embeddings


class SentenceTransformerEncoder(SemanticEncoder):
    """Sentence-Transformers based encoder.

    This module is optional and only required when generating semantic cache.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Failed to import sentence-transformers stack. "
                "Please ensure `sentence-transformers` and HuggingFace `datasets` are installed, "
                f"original error: {exc}"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        prompts: Sequence[str],
        batch_size: int = 16,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Encode prompts with sentence-transformers backend."""
        embeddings = self.model.encode(
            list(prompts),
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata column names."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def run_offline_encoding(
    metadata_csv: Path,
    out_file: Path,
    model_name: str,
    prompts_out_file: Path | None = None,
    batch_size: int = 16,
    normalize_embeddings: bool = False,
) -> None:
    """Generate and save semantic embeddings from node metadata."""
    df = _normalize_columns(pd.read_csv(metadata_csv))
    if "node_index" in df.columns:
        # Keep embeddings aligned with node index order.
        df = df.sort_values("node_index", kind="stable").reset_index(drop=True)

    metas = df.to_dict(orient="records")
    prompts = build_prompts(metas)

    encoder = SentenceTransformerEncoder(model_name=model_name)
    z_sem = encoder.encode(
        prompts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    save_semantic_embeddings(out_file, z_sem)

    if prompts_out_file is not None:
        prompts_out_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prompt": prompts}).to_csv(prompts_out_file, index=False)
        print(f"Saved prompts to: {prompts_out_file}")

    print(f"Saved semantic embeddings: {out_file}")
    print(f"Shape: {z_sem.shape}")
    print(f"Metadata columns: {list(df.columns)}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Offline semantic encoder")
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--out_file", type=Path, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--prompts_out_file",
        type=Path,
        default=None,
        help="Optional CSV path to save generated prompts for inspection.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_offline_encoding(
        metadata_csv=args.metadata_csv,
        out_file=args.out_file,
        model_name=args.model_name,
        prompts_out_file=args.prompts_out_file,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )
