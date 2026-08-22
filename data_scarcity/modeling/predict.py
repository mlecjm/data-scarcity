import csv
import json
import math
from pathlib import Path

from loguru import logger
import typer

from data_scarcity.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "experimental_model.json",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    metrics_path: Path = PROCESSED_DATA_DIR / "test_metrics.json",
    summary_csv_path: Path = PROCESSED_DATA_DIR / "article_results_single_run.csv",
    alpha: float = 0.1,
):
    """Run inference and evaluate deterministic and probabilistic forecasting metrics."""
    if not features_path.exists():
        alt = PROCESSED_DATA_DIR / "target_test_features.csv"
        if alt.exists() and features_path.name == "test_features.csv":
            features_path = alt
        else:
            raise FileNotFoundError(f"Features file not found: {features_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not (0 < alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")

    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    rows = _read_rows(features_path)
    if not rows:
        raise ValueError("No rows to predict")

    residual_quantiles = {
        float(k): float(v) for k, v in model["transfer_residual_quantiles"].items()
    }
    feature_columns: list[str] = list(model["feature_columns"])

    out_rows: list[dict[str, str]] = []
    y_true: list[float] = []
    y_baseline: list[float] = []
    y_target: list[float] = []
    y_transfer: list[float] = []
    q_pred: dict[float, list[float]] = {q: [] for q in sorted(residual_quantiles)}

    for row in rows:
        x = [float(row[col]) for col in feature_columns]
        y = float(row["y"])

        b = _predict(model["baseline"], x)
        t = _predict(model["target_only"], x)
        tr = _predict(model["transfer"], x)

        y_true.append(y)
        y_baseline.append(b)
        y_target.append(t)
        y_transfer.append(tr)

        row_out = {
            "series_id": row.get("series_id", ""),
            "timestamp": row.get("timestamp", ""),
            "y": str(y),
            "yhat_baseline": str(b),
            "yhat_target_only": str(t),
            "yhat_transfer": str(tr),
        }

        for q, rq in residual_quantiles.items():
            qv = tr + rq
            q_pred[q].append(qv)
            row_out[f"q_{q}"] = str(qv)

        out_rows.append(row_out)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(out_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    metrics = {
        "n_test": len(y_true),
        "mae_baseline": _mae(y_true, y_baseline),
        "rmse_baseline": _rmse(y_true, y_baseline),
        "mae_target_only": _mae(y_true, y_target),
        "rmse_target_only": _rmse(y_true, y_target),
        "mae_transfer": _mae(y_true, y_transfer),
        "rmse_transfer": _rmse(y_true, y_transfer),
    }

    levels = sorted(q_pred)
    metrics["pinball_loss_transfer"] = _pinball_multi(y_true, q_pred)

    lo = _closest_level(levels, alpha)
    hi = _closest_level(levels, 1 - alpha)
    lower = q_pred[lo]
    upper = q_pred[hi]
    coverage = sum(
        1
        for y_val, lower_bound, upper_bound in zip(y_true, lower, upper)
        if lower_bound <= y_val <= upper_bound
    ) / len(y_true)
    width = sum(
        upper_bound - lower_bound for lower_bound, upper_bound in zip(lower, upper)
    ) / len(y_true)

    metrics["interval_alpha"] = alpha
    metrics["interval_lower_quantile"] = lo
    metrics["interval_upper_quantile"] = hi
    metrics["interval_coverage"] = coverage
    metrics["interval_avg_width"] = width

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _write_single_run_summary(summary_csv_path, metrics)

    logger.success(
        "Inference complete. Predictions: {} | Metrics: {}",
        predictions_path,
        metrics_path,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _predict(params: dict[str, float], x: list[float]) -> float:
    w = [float(v) for v in params["weights"]]
    return w[0] + sum(wi * xi for wi, xi in zip(w[1:], x))


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def _pinball(y: float, qhat: float, tau: float) -> float:
    u = y - qhat
    return u * (tau - (1.0 if u < 0 else 0.0))


def _pinball_multi(y_true: list[float], q_preds: dict[float, list[float]]) -> float:
    total = 0.0
    n = len(y_true)
    levels = sorted(q_preds)
    for q in levels:
        qh = q_preds[q]
        total += sum(_pinball(y, p, q) for y, p in zip(y_true, qh)) / n
    return total / len(levels)


def _closest_level(levels: list[float], target: float) -> float:
    return min(levels, key=lambda q: abs(q - target))


def _write_single_run_summary(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "baseline",
            "MAE": metrics["mae_baseline"],
            "RMSE": metrics["rmse_baseline"],
        },
        {
            "model": "target_only",
            "MAE": metrics["mae_target_only"],
            "RMSE": metrics["rmse_target_only"],
        },
        {
            "model": "transfer",
            "MAE": metrics["mae_transfer"],
            "RMSE": metrics["rmse_transfer"],
        },
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "MAE", "RMSE"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    app()
