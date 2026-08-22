"""Forecast evaluation metrics — point and probabilistic.

All functions accept plain Python sequences or numpy arrays.  They return
floats and never raise on length-0 input (returning ``float('nan')`` instead),
so they can be used inside bootstrap loops without try/except scaffolding.

References
----------
* Pinball / quantile loss.  Koenker (2005), *Quantile Regression*.
* Continuous Ranked Probability Score (CRPS).  Gneiting & Raftery (2007).
* Winkler / interval score.  Gneiting & Raftery (2007), §6.2.
* MASE.  Hyndman & Koehler (2006), *Int. J. Forecasting* 22(4).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def mae(y: ArrayLike, yhat: ArrayLike) -> float:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: ArrayLike, yhat: ArrayLike) -> float:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: ArrayLike, yhat: ArrayLike, eps: float = 1e-8) -> float:
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    if y.size == 0:
        return float("nan")
    denom = (np.abs(y) + np.abs(yhat)) / 2.0 + eps
    return float(np.mean(np.abs(y - yhat) / denom))


def mase(y: ArrayLike, yhat: ArrayLike, y_train: ArrayLike, season: int = 7) -> float:
    """Mean Absolute Scaled Error against a seasonal-naïve in-sample benchmark.

    For intermittent demand at small sample sizes the scale denominator can
    collapse to zero; we fall back to ``mae(y, yhat)`` in that case to avoid
    inf and document the substitution by returning a value comparable across
    rows in a results table.
    """
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if y.size == 0 or y_train.size <= season:
        return float("nan")
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if denom < 1e-12:
        denom = max(np.mean(np.abs(np.diff(y_train))), 1e-12)
    return float(np.mean(np.abs(y - yhat)) / denom)


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------
def pinball_loss(y: ArrayLike, qhat: ArrayLike, tau: float) -> float:
    y = np.asarray(y, dtype=float); qhat = np.asarray(qhat, dtype=float)
    if y.size == 0:
        return float("nan")
    u = y - qhat
    return float(np.mean(np.maximum(tau * u, (tau - 1.0) * u)))


def multi_pinball_loss(y: ArrayLike, q_preds: Mapping[float, ArrayLike]) -> float:
    levels = sorted(q_preds)
    if not levels:
        return float("nan")
    return float(np.mean([pinball_loss(y, q_preds[t], t) for t in levels]))


def interval_coverage(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    y = np.asarray(y, dtype=float); lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((lo <= y) & (y <= hi)))


def interval_width(lo: ArrayLike, hi: ArrayLike) -> float:
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    if lo.size == 0:
        return float("nan")
    return float(np.mean(hi - lo))


def winkler_score(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, alpha: float) -> float:
    """Winkler / interval score for a central (1-alpha) interval.

    Lower is better.  Captures both width and miscoverage in a single number.
    """
    y = np.asarray(y, dtype=float); lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    if y.size == 0:
        return float("nan")
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    penalty = (2.0 / alpha) * (below + above)
    return float(np.mean(width + penalty))


def crps_from_quantiles(y: ArrayLike, q_preds: Mapping[float, ArrayLike]) -> float:
    """Discrete-quantile CRPS approximation.

    With a grid of K quantile levels ``tau_1<...<tau_K`` and predictions
    ``q_k(x)``, CRPS is well approximated by ``(2/K) * sum_k PB(y, q_k, tau_k)``
    when the levels are equally spaced (Laio & Tamea, 2007).  We renormalise
    by ``2 * mean(PB)`` for arbitrary level grids — a standard approximation
    used in probabilistic forecasting benchmarks.
    """
    levels = sorted(q_preds)
    if not levels:
        return float("nan")
    pb = [pinball_loss(y, q_preds[t], t) for t in levels]
    return float(2.0 * np.mean(pb))


# ---------------------------------------------------------------------------
# Resampling / inference
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(
    y: ArrayLike,
    yhat_a: ArrayLike,
    yhat_b: ArrayLike,
    metric=rmse,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% CI for the difference ``metric(A) - metric(B)``.

    Positive ``delta_mean`` means A is worse than B (for RMSE-like losses).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    yhat_a = np.asarray(yhat_a, dtype=float)
    yhat_b = np.asarray(yhat_b, dtype=float)
    n = y.size
    if n < 2:
        return {"delta_mean": float("nan"), "delta_lo": float("nan"), "delta_hi": float("nan")}

    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[k] = metric(y[idx], yhat_a[idx]) - metric(y[idx], yhat_b[idx])

    return {
        "delta_mean": float(np.mean(deltas)),
        "delta_lo": float(np.quantile(deltas, alpha / 2)),
        "delta_hi": float(np.quantile(deltas, 1 - alpha / 2)),
    }
