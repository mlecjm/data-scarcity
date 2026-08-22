import csv
import json
from pathlib import Path

from loguru import logger
import typer

from data_scarcity.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    source_features_path: Path = PROCESSED_DATA_DIR / "source_features.csv",
    target_train_features_path: Path = PROCESSED_DATA_DIR / "target_train_features.csv",
    model_path: Path = MODELS_DIR / "experimental_model.json",
    transfer_lambda: float = 10.0,
    ridge_lambda: float = 1e-3,
    quantiles: str = "0.1,0.5,0.9",
):
    """Train baseline, target-only, and transfer models with probabilistic residual quantiles."""
    source = _read_xy(source_features_path)
    target = _read_xy(target_train_features_path)

    feature_cols_s, x_source, y_source = source
    feature_cols_t, x_target, y_target = target

    if feature_cols_s != feature_cols_t:
        raise ValueError("Source and target features do not match")
    feature_cols = feature_cols_s

    if len(x_source) < 2 or len(x_target) < 2:
        raise ValueError(
            "Need at least 2 rows in source and target train to fit models"
        )

    source_w = _fit_linear_ridge(x_source, y_source, ridge=ridge_lambda)
    target_w = _fit_linear_ridge(x_target, y_target, ridge=ridge_lambda)
    transfer_w = _fit_linear_transfer(
        x=x_target,
        y=y_target,
        prior=source_w,
        transfer_lambda=transfer_lambda,
        ridge_lambda=ridge_lambda,
    )

    # Baseline is naive forecast y_hat = lag_1.
    baseline_w = [0.0] + [1.0] + [0.0] * (len(feature_cols) - 1)

    q_levels = sorted({float(q.strip()) for q in quantiles.split(",") if q.strip()})
    for q in q_levels:
        if not 0 < q < 1:
            raise ValueError("All quantiles must be in (0,1)")

    transfer_residuals = [
        y - _predict_row(transfer_w, x) for x, y in zip(x_target, y_target)
    ]
    residual_quantiles = {str(q): _quantile(transfer_residuals, q) for q in q_levels}

    model = {
        "feature_columns": feature_cols,
        "baseline": {"weights": baseline_w},
        "source": {"weights": source_w},
        "target_only": {"weights": target_w},
        "transfer": {
            "weights": transfer_w,
            "lambda": transfer_lambda,
            "ridge_lambda": ridge_lambda,
        },
        "quantiles": q_levels,
        "transfer_residual_quantiles": residual_quantiles,
        "train_sizes": {
            "source": len(x_source),
            "target": len(x_target),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    logger.success(
        "Training complete. Model saved to {} (source rows: {}, target rows: {})",
        model_path,
        len(x_source),
        len(x_target),
    )


def _read_xy(path: Path) -> tuple[list[float], list[float]]:
    """Read lag features as matrix X and target y.

    Returns (feature_columns, X, y) where each row in X contains lag values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    x: list[list[float]] = []
    ys: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        feature_cols = sorted(
            [c for c in fieldnames if c.startswith("lag_")], key=_lag_sort_key
        )
        if not feature_cols:
            raise ValueError(f"No lag features found in {path}")

        for row in reader:
            x.append([float(row[col]) for col in feature_cols])
            ys.append(float(row["y"]))
    return feature_cols, x, ys


def _fit_linear_ridge(
    x: list[list[float]], y: list[float], ridge: float
) -> list[float]:
    phi = [[1.0, *row] for row in x]
    p = len(phi[0])

    a = [[0.0 for _ in range(p)] for _ in range(p)]
    b = [0.0 for _ in range(p)]

    for r, yi in zip(phi, y):
        for i in range(p):
            b[i] += r[i] * yi
            for j in range(p):
                a[i][j] += r[i] * r[j]

    # Do not regularize bias term.
    for i in range(1, p):
        a[i][i] += ridge

    return _solve_linear_system(a, b)


def _fit_linear_transfer(
    x: list[list[float]],
    y: list[float],
    prior: list[float],
    transfer_lambda: float,
    ridge_lambda: float,
) -> list[float]:
    phi = [[1.0, *row] for row in x]
    p = len(phi[0])

    a = [[0.0 for _ in range(p)] for _ in range(p)]
    b = [0.0 for _ in range(p)]

    for r, yi in zip(phi, y):
        for i in range(p):
            b[i] += r[i] * yi
            for j in range(p):
                a[i][j] += r[i] * r[j]

    for i in range(p):
        if i > 0:
            a[i][i] += ridge_lambda
        a[i][i] += transfer_lambda
        b[i] += transfer_lambda * prior[i]

    return _solve_linear_system(a, b)


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        pivot_val = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pivot_val

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]

    return [m[i][n] for i in range(n)]


def _predict_row(weights: list[float], x: list[float]) -> float:
    return weights[0] + sum(w * xi for w, xi in zip(weights[1:], x))


def _lag_sort_key(name: str) -> int:
    suffix = name.split("_", maxsplit=1)[1]
    return int(suffix) if suffix.isdigit() else 10**9


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


if __name__ == "__main__":
    app()
