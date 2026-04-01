"""Semantic encoder abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class SemanticEncoder(ABC):
    """Abstract interface for text-to-embedding semantic encoders."""

    @abstractmethod
    def encode(self, prompts: Sequence[str]) -> np.ndarray:
        """Encode prompts into node embeddings.

        Args:
            prompts: Prompt list with length N.

        Returns:
            Embeddings with shape [N, D_sem].
        """
        raise NotImplementedError
