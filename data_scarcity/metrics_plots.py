"""Publication-quality figures for the revised manuscript."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import typer

from data_scarcity.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer(no_args_is_help=False, add_completion=False)


@app.command()
def main(
    model_results_path: Path = PROCESSED_DATA_DIR / "model_comparison_results.csv",
    experiment_grid_path: Path = PROCESSED_DATA_DIR / "experiment_results_grid.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "model_comparison_predictions.csv",
    output_dir: Path = FIGURES_DIR,
    target_alpha: float = 0.2,
    series_model: str = "qr_transfer",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_results_path.exists():
        rows = _read_rows(model_results_path)
        _plot_metric_bars(rows, output_dir / "model_metrics_bars.png")
        _plot_coverage_width(rows, target_alpha, output_dir / "coverage_width_tradeoff.png")
    if experiment_grid_path.exists():
        grid = _read_rows(experiment_grid_path)
        _plot_transfer_gain_heatmap(grid, output_dir / "transfer_gain_heatmap.png")
    if predictions_path.exists():
        _plot_predictive_band(predictions_path, series_model,
                              output_dir / "test_series_confidence_band.png")
    logger.success("Figures written under {}", output_dir)


def _plot_metric_bars(rows, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metric_keys = [("MAE", "MAE"), ("RMSE", "RMSE"),
                   ("pinball_loss", "Pinball loss"), ("winkler", "Winkler")]
    names = [r["name"] for r in rows]
    for ax, (key, title) in zip(axes, metric_keys):
        vals = [float(r[key]) for r in rows]
        bars = ax.bar(names, vals, color="#2563eb", alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Model comparison on target test set (revised pipeline)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_coverage_width(rows, alpha, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        cov = float(r["coverage"]); w = float(r["width"])
        ax.scatter(w, cov, s=80, alpha=0.85)
        ax.annotate(r["name"], (w, cov), fontsize=7, xytext=(4, 4), textcoords="offset points")
    nominal = 1.0 - alpha
    ax.axhline(nominal, color="#dc2626", linestyle="--", linewidth=1.2,
               label=f"Nominal coverage = {nominal:.2f}")
    ax.set_xlabel("Average interval width"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Uncertainty trade-off: coverage vs width")
    ax.legend(loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_transfer_gain_heatmap(rows, out):
    history_vals = sorted({float(r["target_history_fraction"]) for r in rows})
    lambda_vals = sorted({float(r["transfer_lambda"]) for r in rows})
    grid = np.zeros((len(history_vals), len(lambda_vals)))
    for i, h in enumerate(history_vals):
        for j, lam in enumerate(lambda_vals):
            m = next(r for r in rows
                     if float(r["target_history_fraction"]) == h
                     and float(r["transfer_lambda"]) == lam)
            grid[i, j] = float(m["rmse_target_only"]) - float(m["rmse_transfer"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(lambda_vals))); ax.set_yticks(range(len(history_vals)))
    ax.set_xticklabels([f"{v:g}" for v in lambda_vals])
    ax.set_yticklabels([f"{v:.2f}" for v in history_vals])
    ax.set_xlabel(r"Transfer $\lambda$"); ax.set_ylabel("Target history fraction")
    ax.set_title("RMSE gain (positive = transfer better)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:+.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE gain")
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _plot_predictive_band(predictions_path, model, out):
    rows = _read_rows(predictions_path)
    y = np.array([float(r["y"]) for r in rows])
    pred = np.array([float(r[f"{model}_pred"]) for r in rows])
    lo = np.maximum(np.array([float(r[f"{model}_lo"]) for r in rows]), 0.0)
    hi = np.maximum(np.array([float(r[f"{model}_hi"]) for r in rows]), 0.0)
    n = min(60, len(y)); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(x, lo[:n], hi[:n], color="#60a5fa", alpha=0.28,
                    label="10%–90% band (clipped at 0)")
    ax.plot(x, pred[:n], color="#2563eb", linewidth=1.6, label=f"{model} point forecast")
    ax.plot(x, y[:n], color="#111827", linewidth=1.4, label="Observed demand")
    ax.scatter(x, y[:n], color="#111827", s=10)
    ax.axhline(0.0, color="#6b7280", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Test time index"); ax.set_ylabel("Demand")
    ax.set_title(f"Target test segment with predictive band ({model})")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def _read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    app()
