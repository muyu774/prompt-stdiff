"""Data scaler utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Standard scaler with numpy backend.

    The scaler expects traffic tensor shape [T, N, F].
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, mode: str = "per_node") -> "StandardScaler":
        """Fit scaler on training data.

        Args:
            data: Training data in shape [T, N, F].

        Returns:
            Fitted StandardScaler.
        """
        if mode == "global":
            # [1, 1, F]
            mean = data.mean(axis=(0, 1), keepdims=True)
            std = data.std(axis=(0, 1), keepdims=True)
        elif mode == "per_node":
            # [1, N, F]
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unsupported scaler mode: {mode}")
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse standardization."""
        return data * self.std + self.mean
