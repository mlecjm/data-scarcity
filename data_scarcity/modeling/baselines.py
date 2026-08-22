"""Lightweight forecasting baselines for low-data and intermittent demand.

These complement the ridge / random-forest / gradient-boosting baselines.
They are *especially* relevant when reviewers ask for evidence that the
proposed method beats the obvious classical alternatives — a frequent
omission in cold-start forecasting papers.

All baselines expose the same minimal interface::

    yhat = baseline.predict_lagged(x)   # shape (n,)

where ``x`` is the lag-feature matrix produced by ``features.py`` so that the
caller does not need to materialise per-series sequences.

Implementations are numpy-only and avoid stateful imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NaiveLag1:
    """yhat = lag_1."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NaiveLag1":
        return self

    def predict_lagged(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)[:, 0]


@dataclass
class SeasonalNaiveLag7:
    """yhat = lag_7 (matches the standard daily-retail benchmark)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SeasonalNaiveLag7":
        return self

    def predict_lagged(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[1] < 7:
            return x[:, 0]
        return x[:, 6]


@dataclass
class LagEWMA:
    """Exponentially-weighted average over available lags; alpha learned by MAE."""

    alpha: float = 0.4

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LagEWMA":
        # 1-D line search over alpha in [0.05, 0.95]
        best, best_loss = 0.4, float("inf")
        for a in np.linspace(0.05, 0.95, 19):
            self.alpha = float(a)
            loss = float(np.mean(np.abs(y - self.predict_lagged(x))))
            if loss < best_loss:
                best, best_loss = float(a), loss
        self.alpha = best
        return self

    def predict_lagged(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        K = x.shape[1]
        # weights: alpha, alpha*(1-alpha), ..., (1-alpha)^(K-1) — favours recent lags
        w = self.alpha * (1.0 - self.alpha) ** np.arange(K)
        w = w / w.sum()
        return x @ w


@dataclass
class CrostonClassic:
    """Croston (1972) for intermittent demand on a single series of lags.

    Uses lag_1..lag_K as if it were the most recent past series of length K
    and outputs a constant per-row forecast.  This is a *pragmatic* adapter
    for the lag-feature interface used elsewhere in the pipeline.
    """

    alpha: float = 0.1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CrostonClassic":
        return self

    def predict_lagged(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            series = x[i, ::-1]  # oldest -> newest
            nz = series[series > 0]
            if nz.size == 0:
                out[i] = 0.0
                continue
            # demand level: EWMA over non-zero demands
            z = nz[0]
            for v in nz[1:]:
                z = self.alpha * v + (1 - self.alpha) * z
            # interval estimate
            gaps = np.diff(np.where(series > 0)[0]) if (series > 0).sum() > 1 else np.array([1.0])
            p = float(np.mean(gaps))
            out[i] = z / max(p, 1.0)
        return out
