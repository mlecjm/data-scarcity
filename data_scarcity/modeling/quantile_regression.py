"""Linear quantile regression by pinball-loss minimisation.

This module replaces the original ``point + residual-quantile shift`` approach,
which produced *one symmetric* uncertainty band that did not depend on the
input ``x``.  Here we fit one set of weights *per quantile level* so that the
band is genuinely conditional:

    qhat_tau(x) = w_tau . x

We use a vectorised mini-batch subgradient descent (no scipy LP dependency)
with elastic-net-style L2 regularisation and an optional Tibshirani-style
prior term ``lambda_tl * ||w - w_source||^2`` for transfer learning.

Convergence in 200 epochs is typically sufficient for ≤ 30 features and a few
thousand rows; we expose the schedule via ``QuantileRegConfig`` for sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantileRegConfig:
    """Hyperparameters for ``fit_quantile_linear``."""

    n_epochs: int = 300
    lr: float = 0.05
    batch_size: int = 256
    ridge_alpha: float = 1e-3
    transfer_lambda: float = 0.0  # 0.0 ⇒ no source prior
    seed: int = 42


def fit_quantile_linear(
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    prior: np.ndarray | None = None,
    config: QuantileRegConfig = QuantileRegConfig(),
) -> np.ndarray:
    """Fit a single quantile-tau linear model. Returns ``w`` of shape ``(p+1,)``.

    Parameters
    ----------
    x : (n, p) feature matrix.
    y : (n,) targets.
    tau : quantile level in (0, 1).
    prior : optional (p+1,) prior on weights (transfer learning anchor).
    """
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    rng = np.random.default_rng(config.seed)
    n, p = x.shape
    phi = np.concatenate([np.ones((n, 1)), x], axis=1)  # bias term
    w = np.zeros(p + 1)
    if prior is None:
        prior = np.zeros(p + 1)
    # do not regularise the bias term
    reg_mask = np.ones(p + 1)
    reg_mask[0] = 0.0

    for epoch in range(config.n_epochs):
        idx = rng.permutation(n)
        lr = config.lr / (1.0 + epoch / 50.0)  # decay
        for start in range(0, n, config.batch_size):
            sl = idx[start: start + config.batch_size]
            xb, yb = phi[sl], y[sl]
            resid = yb - xb @ w
            # subgradient of pinball loss wrt residual:
            #   tau if resid > 0 else (tau - 1)
            grad_resid = np.where(resid > 0, -tau, 1.0 - tau)  # d/dw of loss
            grad = (xb.T @ grad_resid) / xb.shape[0]
            grad += config.ridge_alpha * reg_mask * w
            if config.transfer_lambda > 0:
                grad += config.transfer_lambda * reg_mask * (w - prior)
            w -= lr * grad
    return w


def predict_linear(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    phi = np.concatenate([np.ones((n, 1)), x], axis=1)
    return phi @ w


# ---------------------------------------------------------------------------
# Wrapper for multi-quantile predictions
# ---------------------------------------------------------------------------
@dataclass
class MultiQuantileLinear:
    levels: list[float]
    weights: dict[float, np.ndarray]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        levels: list[float],
        prior: np.ndarray | None = None,
        config: QuantileRegConfig = QuantileRegConfig(),
    ) -> "MultiQuantileLinear":
        w = {tau: fit_quantile_linear(x, y, tau, prior=prior, config=config) for tau in sorted(levels)}
        return cls(levels=sorted(levels), weights=w)

    def predict_all(self, x: np.ndarray) -> dict[float, np.ndarray]:
        preds = {tau: predict_linear(w, x) for tau, w in self.weights.items()}
        # enforce monotonicity: q_{tau_i} <= q_{tau_j} for tau_i < tau_j
        ordered = sorted(preds)
        for i in range(1, len(ordered)):
            preds[ordered[i]] = np.maximum(preds[ordered[i]], preds[ordered[i - 1]])
        return preds

    def predict_median(self, x: np.ndarray) -> np.ndarray:
        return predict_linear(self.weights[0.5], x)
