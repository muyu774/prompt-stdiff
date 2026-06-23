"""Compatibility exports for semantic-augmented baseline experiments.

This module exposes the fair-comparison semantic adapters under the shorter
``semantic_injection`` name requested by the T-ITS revision protocol. The
implementation lives in ``models.semantic_adapter`` so Prompt-STDiff and
external baseline wrappers share one source of truth.
"""

from __future__ import annotations

from models.semantic_adapter import (
    BatchSemanticComposer,
    DeterministicSemanticInputAdapter,
    DiffusionSemanticConditionAdapter,
    SemanticAdapterConfig,
    build_semantic_adapter_config,
    maybe_build_batch_semantic_composer,
)
from baselines.deterministic_semantic import (
    AGCRNSemanticWrapper,
    DeterministicSemanticBaselineWrapper,
    GWNetSemanticWrapper,
    PDFormerSemanticWrapper,
    adjusted_input_dim,
    deterministic_semantic_extra_dim,
    permute_layout,
)

__all__ = [
    "AGCRNSemanticWrapper",
    "BatchSemanticComposer",
    "DeterministicSemanticInputAdapter",
    "DeterministicSemanticBaselineWrapper",
    "DiffusionSemanticConditionAdapter",
    "GWNetSemanticWrapper",
    "PDFormerSemanticWrapper",
    "SemanticAdapterConfig",
    "adjusted_input_dim",
    "build_semantic_adapter_config",
    "deterministic_semantic_extra_dim",
    "maybe_build_batch_semantic_composer",
    "permute_layout",
]
