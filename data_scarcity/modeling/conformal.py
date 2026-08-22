"""Split-conformal calibration of predictive intervals.

Addresses the under-coverage observed with naïve residual-quantile bands
(0.79 vs nominal 0.80).  Split-conformal calibration yields a *distribution-free*
finite-sample coverage guarantee of approximately ``1 - alpha`` (Vovk et al.,
2005; Lei et al., 2018):

    P( y_test ∈ [qhat_alpha(x) - q_hat, qhat_{1-alpha}(x) + q_hat] ) >= 1 - alpha - 1/(n_cal+1)

where ``q_hat`` is the ``ceil((n_cal+1)(1-alpha))/n_cal``-th quantile of the
calibration nonconformity scores.

We implement the CQR variant (Romano et al., 2019) which conformalises the
*quantile* output rather than a point prediction, so the interval width
remains conditional on ``x``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ConformalAdjustment:
    """Calibration offset for a central (1 - 2*alpha) CQR interval."""

    alpha: float
    qhat: float

    def adjust(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return lo - self.qhat, hi + self.qhat


def fit_cqr_offset(
    y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray, alpha: float
) -> ConformalAdjustment:
    """Compute the conformal adjustment for CQR (Romano et al., 2019)."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if y_cal.size != lo_cal.size or y_cal.size != hi_cal.size:
        raise ValueError("y_cal, lo_cal, hi_cal must have the same length")
    if y_cal.size < 2:
        return ConformalAdjustment(alpha=alpha, qhat=0.0)

    # nonconformity score: max( lo - y, y - hi )
    e = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = e.size
    # quantile level that yields coverage >= 1 - alpha
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    qhat = float(np.sort(e)[k - 1])
    return ConformalAdjustment(alpha=alpha, qhat=qhat)
