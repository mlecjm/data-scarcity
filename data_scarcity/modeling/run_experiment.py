import csv
import json
from pathlib import Path

from loguru import logger
import typer

from data_scarcity.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from data_scarcity.dataset import main as dataset_main
from data_scarcity.features import main as features_main
from data_scarcity.modeling.predict import main as predict_main
from data_scarcity.modeling.train import main as train_main

app = typer.Typer()


@app.command()
def main(
    sales_input_path: Path = RAW_DATA_DIR / "m5" / "sales_train_validation.csv",
    results_csv_path: Path = PROCESSED_DATA_DIR / "experiment_results_grid.csv",
    target_ratio: float = 0.2,
    history_grid: str = "0.1,0.2,0.3",
    lambda_grid: str = "1,10,50",
    lag_count: int = 7,
    target_test_fraction: float = 0.3,
    quantiles: str = "0.1,0.5,0.9",
    alpha: float = 0.1,
    max_series: int = 250,
    max_days: int = 365,
    seed: int = 42,
):
    """Run grid experiments and export a comparison table for article-ready results."""
    history_values = [float(v.strip()) for v in history_grid.split(",") if v.strip()]
    lambda_values = [float(v.strip()) for v in lambda_grid.split(",") if v.strip()]

    all_rows: list[dict[str, float | int]] = []

    for history_frac in history_values:
        for lam in lambda_values:
            logger.info(
                "Running experiment with target_history_fraction={} and lambda={}",
                history_frac,
                lam,
            )

            dataset_main(
                input_path=sales_input_path,
                target_ratio=target_ratio,
                target_history_fraction=history_frac,
                max_series=max_series,
                max_days=max_days,
                seed=seed,
            )

            features_main(
                lag_count=lag_count,
                target_test_fraction=target_test_fraction,
            )

            train_main(
                transfer_lambda=lam,
                quantiles=quantiles,
                model_path=MODELS_DIR / "experimental_model.json",
            )

            predict_main(
                features_path=PROCESSED_DATA_DIR / "target_test_features.csv",
                model_path=MODELS_DIR / "experimental_model.json",
                predictions_path=PROCESSED_DATA_DIR / "test_predictions.csv",
                metrics_path=PROCESSED_DATA_DIR / "test_metrics.json",
                alpha=alpha,
            )

            metrics = _read_json(PROCESSED_DATA_DIR / "test_metrics.json")
            row = {
                "target_history_fraction": history_frac,
                "transfer_lambda": lam,
                "n_test": int(metrics["n_test"]),
                "mae_baseline": float(metrics["mae_baseline"]),
                "mae_target_only": float(metrics["mae_target_only"]),
                "mae_transfer": float(metrics["mae_transfer"]),
                "rmse_baseline": float(metrics["rmse_baseline"]),
                "rmse_target_only": float(metrics["rmse_target_only"]),
                "rmse_transfer": float(metrics["rmse_transfer"]),
                "pinball_loss_transfer": float(metrics["pinball_loss_transfer"]),
                "interval_coverage": float(metrics["interval_coverage"]),
                "interval_avg_width": float(metrics["interval_avg_width"]),
            }
            all_rows.append(row)

    _write_csv(results_csv_path, all_rows)
    logger.success("Grid experiments complete. Results saved to {}", results_csv_path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("No result rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    app()
