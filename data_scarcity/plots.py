from collections import defaultdict
import csv
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import typer

from data_scarcity.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "target_test_features.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "model_comparison_predictions.csv",
    output_path: Path = FIGURES_DIR / "test_series_confidence_band.png",
    model_name: str = "ridge_transfer",
    series_id: str | None = None,
):
    """Plot a real target-test series with 10%-90% confidence band."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    feature_rows = _read_rows(features_path)
    prediction_rows = _read_rows(predictions_path)
    if len(feature_rows) != len(prediction_rows):
        raise ValueError("Features and predictions must have the same number of rows")

    aligned_rows = []
    for feature_row, prediction_row in zip(feature_rows, prediction_rows):
        if feature_row.get("y") != prediction_row.get("y"):
            # Keep going but retain the aligned row structure. The order is expected to match.
            logger.warning(
                "Target row mismatch detected between features and predictions"
            )
        aligned_rows.append((feature_row, prediction_row))

    chosen_series = series_id or _select_series(aligned_rows)
    series_rows = [
        (feature_row, prediction_row)
        for feature_row, prediction_row in aligned_rows
        if feature_row.get("series_id") == chosen_series
    ]
    if not series_rows:
        raise ValueError(f"No rows found for series_id={chosen_series}")

    ordered_rows = sorted(
        series_rows, key=lambda pair: _time_sort_key(pair[0].get("timestamp", ""))
    )
    x_values = list(range(len(ordered_rows)))
    y_true = [float(feature_row["y"]) for feature_row, _ in ordered_rows]
    q10 = [
        float(prediction_row[f"{model_name}_lo"]) for _, prediction_row in ordered_rows
    ]
    q90 = [
        float(prediction_row[f"{model_name}_hi"]) for _, prediction_row in ordered_rows
    ]
    y_pred = [
        float(prediction_row[f"{model_name}_pred"])
        for _, prediction_row in ordered_rows
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(x_values, y_true, label="Observed demand", color="#1f2937", linewidth=2)
    plt.plot(
        x_values,
        y_pred,
        label=f"{model_name} point forecast",
        color="#2563eb",
        linewidth=2,
    )
    plt.fill_between(
        x_values, q10, q90, color="#60a5fa", alpha=0.25, label="10%-90% band"
    )
    plt.scatter(x_values, y_true, color="#111827", s=12, alpha=0.65)
    plt.title(f"Target test series with predictive band ({chosen_series})")
    plt.xlabel("Test time index")
    plt.ylabel("Demand")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success("Confidence band plot saved to {}", output_path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select_series(aligned_rows: list[tuple[dict[str, str], dict[str, str]]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for feature_row, _ in aligned_rows:
        counts[feature_row.get("series_id", "")] += 1
    return max(counts, key=counts.get)


def _time_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("d_"):
        suffix = value[2:]
        if suffix.isdigit():
            return (int(suffix), value)
    try:
        return (int(float(value)), value)
    except ValueError:
        return (0, value)


if __name__ == "__main__":
    app()
