"""Per-column z-score statistics.

Actions and oracle states are whitened before they reach the model, which keeps
the action encoder's inputs on a sane scale. Statistics are estimated once from
a subsample of rows and cached next to the dataset, so training and every
analysis script share the exact same numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Normalizer:
    """Holds `{column: (mean, std)}` and applies them to arrays or tensors."""

    def __init__(self, stats: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.stats = {k: (np.asarray(m, np.float32), np.asarray(s, np.float32)) for k, (m, s) in stats.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.stats

    def normalize(self, key: str, x):
        if key not in self.stats:
            return x
        mean, std = self._as(x, key)
        return (x - mean) / std

    def denormalize(self, key: str, x):
        if key not in self.stats:
            return x
        mean, std = self._as(x, key)
        return x * std + mean

    def _as(self, x, key: str):
        mean, std = self.stats[key]
        if torch.is_tensor(x):
            return (
                torch.as_tensor(mean, dtype=x.dtype, device=x.device),
                torch.as_tensor(std, dtype=x.dtype, device=x.device),
            )
        return mean, std

    def dim(self, key: str) -> int:
        return int(self.stats[key][0].shape[-1])

    # ---- persistence ---------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for key, (mean, std) in self.stats.items():
            flat[f"{key}.mean"], flat[f"{key}.std"] = mean, std
        np.savez(path, **flat)

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        data = np.load(path)
        stats: dict[str, list] = {}
        for name in data.files:
            key, kind = name.rsplit(".", 1)
            stats.setdefault(key, [None, None])[0 if kind == "mean" else 1] = data[name]
        return cls({k: tuple(v) for k, v in stats.items()})

    @classmethod
    def from_columns(cls, columns: dict[str, np.ndarray], eps: float = 1e-6) -> "Normalizer":
        """Estimate mean/std per column, ignoring rows with NaNs."""
        stats = {}
        for key, values in columns.items():
            values = np.asarray(values, dtype=np.float64)
            if values.ndim == 1:
                values = values[:, None]
            values = values[~np.isnan(values).any(axis=1)]
            if len(values) == 0:
                raise ValueError(f"column {key!r} is all NaN")
            std = values.std(0)
            # A constant column would otherwise blow up to +-inf.
            stats[key] = (values.mean(0).astype(np.float32), np.maximum(std, eps).astype(np.float32))
        return cls(stats)
