"""Unit tests for the revised pipeline.

Run with ``pytest -q``.  Tests are written to run offline (no M5 download
required) using ``scripts/make_synthetic_m5.py`` to generate a small fixture.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from data_scarcity import config
from data_scarcity.metrics import (
    crps_from_quantiles,
    interval_coverage,
    mae,
    mase,
    multi_pinball_loss,
    paired_bootstrap_ci,
    pinball_loss,
    rmse,
    smape,
    winkler_score,
)
from data_scarcity.modeling.baselines import CrostonClassic, LagEWMA, NaiveLag1, SeasonalNaiveLag7
from data_scarcity.modeling.conformal import fit_cqr_offset
from data_scarcity.modeling.quantile_regression import (
    MultiQuantileLinear,
    QuantileRegConfig,
    fit_quantile_linear,
    predict_linear,
)
from data_scarcity.decisions import (
    NewsvendorConfig,
    expected_cost,
    newsvendor_orders,
    safety_stock,
    service_level,
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_config_paths_are_initialized():
    assert isinstance(config.PROJ_ROOT, Path)
    assert config.DATA_DIR == config.PROJ_ROOT / "data"
    assert config.MODELS_DIR == config.PROJ_ROOT / "models"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_mae_rmse_basic(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
        assert mae([1.0, 3.0], [2.0, 0.0]) == 2.0

    def test_pinball_loss_median_equals_half_abs(self):
        y = np.array([0.0, 1.0, 2.0]); q = np.array([1.0, 1.0, 1.0])
        # for tau = 0.5, pinball loss = 0.5 * mean |y - q|
        assert pinball_loss(y, q, 0.5) == pytest.approx(0.5 * mae(y, q))

    def test_pinball_loss_extreme_quantiles(self):
        # Asymmetric residuals: model under-predicts → tau=0.9 should penalise more
        y = np.array([1.0, 1.0, 1.0]); q = np.array([0.0, 0.0, 0.0])
        loss_low = pinball_loss(y, q, 0.1)
        loss_high = pinball_loss(y, q, 0.9)
        assert loss_high > loss_low

    def test_smape_zero_on_perfect(self):
        assert smape([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0, abs=1e-6)

    def test_mase_falls_back_when_seasonal_zero(self):
        # constant series → seasonal-naive denominator is 0
        v = mase([1, 2, 3], [1, 2, 3], y_train=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert np.isfinite(v)

    def test_interval_coverage_and_winkler(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5]); hi = np.array([1.5, 2.5, 3.5])
        assert interval_coverage(y, lo, hi) == 1.0
        assert winkler_score(y, lo, hi, alpha=0.2) == pytest.approx(1.0)

    def test_crps_monotone_in_pinball(self):
        y = np.array([1.0, 2.0])
        preds = {0.1: np.array([0.8, 1.8]), 0.5: np.array([1.0, 2.0]), 0.9: np.array([1.2, 2.2])}
        c1 = crps_from_quantiles(y, preds)
        # double the spread → larger CRPS
        preds2 = {0.1: np.array([0.6, 1.6]), 0.5: np.array([1.0, 2.0]), 0.9: np.array([1.4, 2.4])}
        c2 = crps_from_quantiles(y, preds2)
        assert c2 > c1

    def test_bootstrap_ci_zero_when_identical(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        yhat = rng.normal(size=200)
        ci = paired_bootstrap_ci(y, yhat, yhat, metric=mae, n_boot=200)
        assert ci["delta_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# quantile regression
# ---------------------------------------------------------------------------
class TestQuantileRegression:
    def test_recovers_linear_signal(self):
        rng = np.random.default_rng(0)
        n, p = 800, 4
        X = rng.normal(size=(n, p))
        w_true = np.array([0.5, -1.0, 0.3, 0.0])
        y = X @ w_true + rng.normal(scale=0.1, size=n)
        w = fit_quantile_linear(X, y, tau=0.5, config=QuantileRegConfig(n_epochs=400, lr=0.05))
        # bias + 4 coefficients ⇒ 5
        assert w.shape == (5,)
        # coefficients should be close
        assert np.allclose(w[1:], w_true, atol=0.2)

    def test_multi_quantile_monotone(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(400, 3))
        y = X.sum(axis=1) + rng.normal(size=400)
        mq = MultiQuantileLinear.fit(X, y, levels=[0.1, 0.5, 0.9])
        preds = mq.predict_all(X[:10])
        assert np.all(preds[0.1] <= preds[0.5])
        assert np.all(preds[0.5] <= preds[0.9])


# ---------------------------------------------------------------------------
# conformal
# ---------------------------------------------------------------------------
def test_cqr_brings_coverage_up_to_target():
    rng = np.random.default_rng(2)
    n = 500
    y = rng.normal(size=n)
    # initially under-cover: tight intervals
    lo = y - 0.1 + rng.normal(scale=0.05, size=n)
    hi = y + 0.1 + rng.normal(scale=0.05, size=n)
    cqr = fit_cqr_offset(y, lo, hi, alpha=0.1)
    new_lo, new_hi = cqr.adjust(lo, hi)
    cov = float(np.mean((new_lo <= y) & (y <= new_hi)))
    assert cov >= 0.85   # at least near nominal 0.9


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
class TestBaselines:
    def setup_method(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(50, 7))
        self.y = self.X[:, 0] + rng.normal(scale=0.1, size=50)

    def test_naive_lag1_returns_lag1(self):
        out = NaiveLag1().fit(self.X, self.y).predict_lagged(self.X)
        assert np.allclose(out, self.X[:, 0])

    def test_seasonal_lag7(self):
        out = SeasonalNaiveLag7().fit(self.X, self.y).predict_lagged(self.X)
        assert np.allclose(out, self.X[:, 6])

    def test_ewma_learns_an_alpha(self):
        m = LagEWMA().fit(self.X, self.y)
        assert 0.0 < m.alpha < 1.0

    def test_croston_handles_zero_history(self):
        zeros = np.zeros((5, 7))
        out = CrostonClassic().predict_lagged(zeros)
        assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# decision layer
# ---------------------------------------------------------------------------
def test_newsvendor_critical_ratio_selects_correct_quantile():
    cfg = NewsvendorConfig(c_u=4.0, c_o=1.0)        # critical ratio = 0.8
    q_preds = {0.1: np.array([1.0]), 0.5: np.array([2.0]), 0.9: np.array([3.0])}
    orders = newsvendor_orders(q_preds, cfg=cfg)
    # nearest to 0.8 is 0.9
    assert orders[0] == 3.0


def test_service_level_and_expected_cost_signs():
    y = np.array([1.0, 5.0, 3.0])
    orders = np.array([2.0, 2.0, 4.0])
    cfg = NewsvendorConfig(c_u=4.0, c_o=1.0)
    sl = service_level(y, orders)
    assert 0.0 < sl < 1.0
    cost = expected_cost(y, orders, cfg=cfg)
    assert cost > 0.0
    ss = safety_stock(orders, point=np.array([1.0, 2.0, 3.0]))
    assert ss >= 0.0


# ---------------------------------------------------------------------------
# end-to-end smoke test (runs the actual CLIs on a tiny synthetic dataset)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "m5.csv"
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "make_synthetic_m5.py"), str(csv_path)],
        check=True,
    )

    env = {"PYTHONPATH": str(repo), "DS_DATA_DIR": str(tmp_path / "data")}
    (tmp_path / "data" / "raw" / "m5").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw" / "m5" / "sales_train_validation.csv").write_bytes(csv_path.read_bytes())

    def run(mod, *args):
        subprocess.run([sys.executable, "-m", mod, *args], check=True, env={**__import__("os").environ, **env}, cwd=repo)

    run("data_scarcity.dataset",
        "--input-path", str(tmp_path / "data" / "raw" / "m5" / "sales_train_validation.csv"),
        "--max-series", "60", "--max-days", "200", "--target-history-fraction", "0.3")
    run("data_scarcity.features", "--lag-count", "5", "--target-test-fraction", "0.3")
    run("data_scarcity.modeling.compare_models", "--transfer-lambda", "10", "--alpha", "0.1")

    metrics_csv = tmp_path / "data" / "processed" / "model_comparison_results.csv"
    assert metrics_csv.exists()
