"""Compare forecasting models on a common low-data target test set.

What's new vs the original ``compare_models.py``:

* Adds true *per-quantile* linear regression (``MultiQuantileLinear``) as the
  probabilistic head, so the predictive band depends on the inputs.
* Adds a calibration split (taken from the *target train* set) used to run
  CQR-style split-conformal calibration of the interval.  This empirically
  fixes the under-coverage of ~0.79 observed with the naïve residual-shift.
* Reports MAE, RMSE, sMAPE, MASE, pinball loss, CRPS approximation, interval
  coverage / width / Winkler score.
* Adds simple naïve, seasonal-naïve, EWMA and Croston baselines.
* Records training and inference latency per model.

Outputs:
    data/processed/model_comparison_results.csv
    data/processed/model_comparison_predictions.csv
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Callable, Mapping

from loguru import logger
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import typer

from data_scarcity.config import GLOBAL_SEED, PROCESSED_DATA_DIR
from data_scarcity.metrics import (
    crps_from_quantiles,
    interval_coverage,
    interval_width,
    mae,
    mase,
    multi_pinball_loss,
    pinball_loss,
    rmse,
    smape,
    winkler_score,
)
from data_scarcity.modeling.baselines import (
    CrostonClassic,
    LagEWMA,
    NaiveLag1,
    SeasonalNaiveLag7,
)
from data_scarcity.modeling.conformal import fit_cqr_offset
from data_scarcity.modeling.quantile_regression import MultiQuantileLinear, QuantileRegConfig

app = typer.Typer(no_args_is_help=False, add_completion=False)


@dataclass
class ModelResult:
    name: str
    MAE: float
    RMSE: float
    sMAPE: float
    MASE: float
    pinball_loss: float
    crps: float
    coverage: float
    width: float
    winkler: float
    train_time_s: float
    inference_time_s: float


@app.command()
def main(
    source_features_path: Path = PROCESSED_DATA_DIR / "source_features.csv",
    target_train_features_path: Path = PROCESSED_DATA_DIR / "target_train_features.csv",
    target_test_features_path: Path = PROCESSED_DATA_DIR / "target_test_features.csv",
    results_csv_path: Path = PROCESSED_DATA_DIR / "model_comparison_results.csv",
    predictions_csv_path: Path = PROCESSED_DATA_DIR / "model_comparison_predictions.csv",
    quantiles: str = "0.1,0.5,0.9",
    ridge_alpha: float = 1e-3,
    transfer_lambda: float = 10.0,
    calibration_fraction: float = 0.2,
    alpha: float = 0.1,
    random_state: int = GLOBAL_SEED,
) -> None:
    """Compare baselines, ridge variants and quantile regressors."""
    levels = sorted({float(q.strip()) for q in quantiles.split(",") if q.strip()})
    if not levels:
        raise ValueError("At least one quantile level required")

    # --------------------------------------------------------------- data ---
    fcols_s, xs, ys = _read_xy(source_features_path)
    fcols_t, xt, yt = _read_xy(target_train_features_path)
    fcols_te, xte, yte = _read_xy(target_test_features_path)
    if not (fcols_s == fcols_t == fcols_te):
        raise ValueError("Feature columns must match across splits")

    # carve a calibration slice from the *end* of the target train set
    n_cal = max(8, int(len(xt) * calibration_fraction))
    xt_fit, yt_fit = xt[:-n_cal], yt[:-n_cal]
    xt_cal, yt_cal = xt[-n_cal:], yt[-n_cal:]
    logger.info(
        "splits | source {} | tgt fit {} | tgt cal {} | tgt test {}",
        len(xs), len(xt_fit), len(xt_cal), len(xte),
    )

    qcfg_base = QuantileRegConfig(seed=random_state, ridge_alpha=ridge_alpha, transfer_lambda=0.0)
    qcfg_tl = QuantileRegConfig(seed=random_state, ridge_alpha=ridge_alpha, transfer_lambda=transfer_lambda)

    # ---------------------------------------------------------------- fits --
    results: list[ModelResult] = []
    pred_table: dict[str, dict[str, np.ndarray]] = {}  # name -> {pred, lo, hi}

    # Baselines
    for name, b in [
        ("naive_lag1", NaiveLag1()),
        ("seasonal_naive_lag7", SeasonalNaiveLag7()),
        ("ewma", LagEWMA()),
        ("croston", CrostonClassic()),
    ]:
        results.append(_score_pointwise(name, b, xt_fit, yt_fit, xt_cal, yt_cal, xte, yte, alpha, levels, pred_table))

    # Ridge (target-only, source-only, transfer-regularised)
    results.append(_score_pointwise("ridge_target_only", Ridge(alpha=ridge_alpha, random_state=random_state),
                                    xt_fit, yt_fit, xt_cal, yt_cal, xte, yte, alpha, levels, pred_table))
    ridge_src = Ridge(alpha=ridge_alpha, random_state=random_state).fit(xs, ys)
    results.append(_score_pointwise("ridge_source_only", _Frozen(ridge_src), xt_fit, yt_fit, xt_cal, yt_cal,
                                    xte, yte, alpha, levels, pred_table))
    transfer_ridge = _fit_transfer_ridge(xt_fit, yt_fit, ridge_src.coef_, ridge_src.intercept_, ridge_alpha, transfer_lambda)
    results.append(_score_pointwise("ridge_transfer", _Frozen(transfer_ridge), xt_fit, yt_fit, xt_cal, yt_cal,
                                    xte, yte, alpha, levels, pred_table))

    # sklearn nonlinear baselines
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2,
                               random_state=random_state, n_jobs=-1)
    results.append(_score_pointwise("random_forest", rf, xt_fit, yt_fit, xt_cal, yt_cal, xte, yte, alpha, levels, pred_table))
    gb = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3,
                                   random_state=random_state, loss="squared_error")
    results.append(_score_pointwise("gradient_boosting", gb, xt_fit, yt_fit, xt_cal, yt_cal, xte, yte, alpha, levels, pred_table))

    # Quantile regression — target-only
    t0 = time.perf_counter()
    qr_target = MultiQuantileLinear.fit(xt_fit, yt_fit, levels=levels, config=qcfg_base)
    train_t = time.perf_counter() - t0
    results.append(_score_quantile("qr_target_only", qr_target, xt_cal, yt_cal, xte, yte, alpha, levels, pred_table, train_t))

    # Quantile regression — source pretrain, target fine-tune w/ regularisation
    qr_src = MultiQuantileLinear.fit(xs, ys, levels=levels, config=qcfg_base)
    prior_w = qr_src.weights[0.5]  # use median weights as anchor
    t0 = time.perf_counter()
    qr_transfer = MultiQuantileLinear.fit(xt_fit, yt_fit, levels=levels, prior=prior_w, config=qcfg_tl)
    train_t = time.perf_counter() - t0
    results.append(_score_quantile("qr_transfer", qr_transfer, xt_cal, yt_cal, xte, yte, alpha, levels, pred_table, train_t))

    # ---------------------------------------------------------------- write --
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with results_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    _write_predictions(predictions_csv_path, fcols_te, xte, yte, pred_table)
    logger.success("Wrote {} and {}", results_csv_path, predictions_csv_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _Frozen:
    """Wrap an already-fitted sklearn estimator so the generic loop reuses it."""

    def __init__(self, est):
        self._est = est

    def fit(self, x, y):
        return self  # already fitted

    def predict(self, x):
        return self._est.predict(x)


def _fit_transfer_ridge(
    x: np.ndarray, y: np.ndarray, prior_coef: np.ndarray, prior_intercept: float,
    ridge_alpha: float, transfer_lambda: float,
) -> Ridge:
    """Closed-form: minimise ||y - Xw||^2 + alpha||w||^2 + lambda||w - w_src||^2."""
    n, p = x.shape
    a = x.T @ x + (ridge_alpha + transfer_lambda) * np.eye(p)
    b = x.T @ (y - prior_intercept) + transfer_lambda * prior_coef
    w = np.linalg.solve(a, b)
    est = Ridge(alpha=ridge_alpha)
    est.coef_ = w
    est.intercept_ = float(prior_intercept)
    return est


def _read_xy(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or [])
                if c not in {"series_id", "timestamp", "y"}]
        xs: list[list[float]] = []
        ys: list[float] = []
        for row in reader:
            xs.append([float(row[c]) for c in cols])
            ys.append(float(row["y"]))
    return cols, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _empirical_residual_quantiles(resid: np.ndarray, levels: list[float]) -> dict[float, float]:
    return {q: float(np.quantile(resid, q)) for q in levels}


def _score_pointwise(
    name: str, estimator, x_fit: np.ndarray, y_fit: np.ndarray,
    x_cal: np.ndarray, y_cal: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    alpha: float, levels: list[float],
    pred_table: dict[str, dict[str, np.ndarray]],
) -> ModelResult:
    t0 = time.perf_counter()
    if hasattr(estimator, "fit") and not isinstance(estimator, _Frozen):
        # baselines may not have .predict; check below
        try:
            estimator.fit(x_fit, y_fit)
        except TypeError:
            estimator.fit(np.asarray(x_fit), np.asarray(y_fit))
    train_t = time.perf_counter() - t0

    predict_fn: Callable[[np.ndarray], np.ndarray]
    if hasattr(estimator, "predict_lagged"):
        predict_fn = estimator.predict_lagged
    else:
        predict_fn = estimator.predict

    t0 = time.perf_counter()
    yhat_cal = predict_fn(x_cal)
    yhat_test = predict_fn(x_test)
    inf_t = time.perf_counter() - t0

    resid_cal = y_cal - yhat_cal
    qmap = _empirical_residual_quantiles(resid_cal, levels)
    lo_q, hi_q = min(levels), max(levels)

    # build quantile predictions and intervals (use calibration-residual shift)
    q_preds_test = {tau: yhat_test + qmap[tau] for tau in levels}
    # Conformal calibration of the (lo_q, hi_q) interval
    lo_cal_pred = yhat_cal + qmap[lo_q]
    hi_cal_pred = yhat_cal + qmap[hi_q]
    cqr = fit_cqr_offset(y_cal, lo_cal_pred, hi_cal_pred, alpha=alpha)
    lo_test = yhat_test + qmap[lo_q]
    hi_test = yhat_test + qmap[hi_q]
    lo_cal_test, hi_cal_test = cqr.adjust(lo_test, hi_test)

    pred_table[name] = {"pred": yhat_test, "lo": lo_cal_test, "hi": hi_cal_test}

    return ModelResult(
        name=name,
        MAE=mae(y_test, yhat_test),
        RMSE=rmse(y_test, yhat_test),
        sMAPE=smape(y_test, yhat_test),
        MASE=mase(y_test, yhat_test, y_fit),
        pinball_loss=multi_pinball_loss(y_test, q_preds_test),
        crps=crps_from_quantiles(y_test, q_preds_test),
        coverage=interval_coverage(y_test, lo_cal_test, hi_cal_test),
        width=interval_width(lo_cal_test, hi_cal_test),
        winkler=winkler_score(y_test, lo_cal_test, hi_cal_test, alpha=2 * alpha),
        train_time_s=train_t,
        inference_time_s=inf_t,
    )


def _score_quantile(
    name: str, qr: MultiQuantileLinear,
    x_cal: np.ndarray, y_cal: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    alpha: float, levels: list[float],
    pred_table: dict[str, dict[str, np.ndarray]],
    train_time_s: float,
) -> ModelResult:
    t0 = time.perf_counter()
    q_test = qr.predict_all(x_test)
    q_cal = qr.predict_all(x_cal)
    inf_t = time.perf_counter() - t0

    yhat_test = qr.predict_median(x_test)
    lo_q, hi_q = min(levels), max(levels)
    cqr = fit_cqr_offset(y_cal, q_cal[lo_q], q_cal[hi_q], alpha=alpha)
    lo_cal_test, hi_cal_test = cqr.adjust(q_test[lo_q], q_test[hi_q])

    pred_table[name] = {"pred": yhat_test, "lo": lo_cal_test, "hi": hi_cal_test}

    return ModelResult(
        name=name,
        MAE=mae(y_test, yhat_test),
        RMSE=rmse(y_test, yhat_test),
        sMAPE=smape(y_test, yhat_test),
        MASE=mase(y_test, yhat_test, y_cal),
        pinball_loss=multi_pinball_loss(y_test, q_test),
        crps=crps_from_quantiles(y_test, q_test),
        coverage=interval_coverage(y_test, lo_cal_test, hi_cal_test),
        width=interval_width(lo_cal_test, hi_cal_test),
        winkler=winkler_score(y_test, lo_cal_test, hi_cal_test, alpha=2 * alpha),
        train_time_s=train_time_s,
        inference_time_s=inf_t,
    )


def _write_predictions(
    path: Path, feature_cols: list[str], x_test: np.ndarray, y_test: np.ndarray,
    pred_table: Mapping[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(pred_table)
    fields = ["y"] + [f"x_{c}" for c in feature_cols]
    for n in names:
        fields.extend([f"{n}_pred", f"{n}_lo", f"{n}_hi"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(y_test)):
            row = {"y": float(y_test[i])}
            for j, c in enumerate(feature_cols):
                row[f"x_{c}"] = float(x_test[i, j])
            for n in names:
                row[f"{n}_pred"] = float(pred_table[n]["pred"][i])
                row[f"{n}_lo"] = float(pred_table[n]["lo"][i])
                row[f"{n}_hi"] = float(pred_table[n]["hi"][i])
            w.writerow(row)


if __name__ == "__main__":
    app()
