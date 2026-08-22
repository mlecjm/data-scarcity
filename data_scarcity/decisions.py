"""Decision layer: from predictive intervals to inventory actions.

Reviewer 3 specifically observed: "Stage 4 promises to map outputs to
stockout vs. overstock decisions ... no inventory cost calculation, no
service level computation, no reorder point analysis, and no safety stock
quantification is presented anywhere."

This module fills that gap.  For each test-step quantile prediction
``qhat_tau`` the newsvendor solution to

    min_{order}  c_u * E[(D - order)+]  +  c_o * E[(order - D)+]

is given by ordering ``qhat_{tau*}`` with ``tau* = c_u / (c_u + c_o)``
(Arrow, Harris & Marschak, 1951).  We expose:

* ``newsvendor_orders(q_preds, c_u, c_o)`` — order quantities
* ``service_level(y, orders)`` — empirical fill rate (no stockout)
* ``expected_cost(y, orders, c_u, c_o)`` — average per-period cost
* ``safety_stock(orders, point)`` — implied safety margin above the median
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class NewsvendorConfig:
    """Per-unit understock (c_u) and overstock (c_o) costs."""

    c_u: float = 4.0    # lost margin / stockout cost per unit
    c_o: float = 1.0    # holding cost per unit per period

    @property
    def critical_ratio(self) -> float:
        return self.c_u / (self.c_u + self.c_o)


def _nearest_quantile(levels: list[float], target: float) -> float:
    return min(levels, key=lambda q: abs(q - target))


def newsvendor_orders(
    q_preds: Mapping[float, np.ndarray], cfg: NewsvendorConfig = NewsvendorConfig(),
    clip_nonneg: bool = True,
) -> np.ndarray:
    """Order qhat at the critical ratio quantile.

    The optimal order quantity for the single-period newsvendor with linear
    under-/over-stock costs is the ``c_u/(c_u+c_o)``-th quantile of the
    predictive distribution.
    """
    if not q_preds:
        raise ValueError("q_preds must contain at least one quantile")
    cr = cfg.critical_ratio
    levels = sorted(q_preds)
    chosen = _nearest_quantile(levels, cr)
    orders = np.asarray(q_preds[chosen], dtype=float).copy()
    if clip_nonneg:
        orders = np.maximum(orders, 0.0)
    return orders


def service_level(y: np.ndarray, orders: np.ndarray) -> float:
    """Empirical Type-I service level (fraction of periods with no stockout)."""
    y = np.asarray(y, dtype=float); orders = np.asarray(orders, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean(orders >= y))


def expected_cost(
    y: np.ndarray, orders: np.ndarray, cfg: NewsvendorConfig = NewsvendorConfig(),
) -> float:
    """Average per-period cost for the realised demand vs the placed orders."""
    y = np.asarray(y, dtype=float); orders = np.asarray(orders, dtype=float)
    if y.size == 0:
        return float("nan")
    under = np.maximum(y - orders, 0.0)
    over = np.maximum(orders - y, 0.0)
    return float(np.mean(cfg.c_u * under + cfg.c_o * over))


def safety_stock(orders: np.ndarray, point: np.ndarray) -> float:
    """Average implied safety stock above the median (point) forecast."""
    orders = np.asarray(orders, dtype=float); point = np.asarray(point, dtype=float)
    if orders.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(orders - point, 0.0)))
