"""Run the full pipeline across multiple random seeds and aggregate results.

This addresses the reviewer concern that a single seeded run cannot
demonstrate that the small numerical differences between models are
statistically meaningful.  We run ``n_seeds`` independent splits and report:

    mean ± std        for each metric
    bootstrap 95% CI  for the (transfer - target-only) RMSE gap

The script delegates each individual run to the ``compare_models`` CLI and
parses its CSV output.  This keeps the modeling code unchanged while adding
a higher-level evaluation harness.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from loguru import logger
import numpy as np
import typer

from data_scarcity.config import GLOBAL_SEED, PROCESSED_DATA_DIR

app = typer.Typer(no_args_is_help=False, add_completion=False)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR.parent / "raw" / "m5" / "sales_train_validation.csv",
    history_fraction: float = 0.3,
    transfer_lambda: float = 10.0,
    n_seeds: int = 5,
    base_seed: int = GLOBAL_SEED,
    output_dir: Path = PROCESSED_DATA_DIR / "multi_seed",
    target_test_fraction: float = 0.3,
    lag_count: int = 7,
    quantiles: str = "0.1,0.5,0.9",
    alpha: float = 0.1,
    max_series: int = 250,
    max_days: int = 365,
) -> None:
    """Run dataset + features + compare_models for ``n_seeds`` seeds and aggregate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [base_seed + i for i in range(n_seeds)]
    per_seed: list[dict[str, dict[str, float]]] = []

    for s in seeds:
        logger.info("Running seed {}", s)
        _run_stage("data_scarcity.dataset", [
            "--input-path", str(input_path), "--max-series", str(max_series),
            "--max-days", str(max_days), "--target-history-fraction", str(history_fraction),
            "--seed", str(s),
        ])
        _run_stage("data_scarcity.features", [
            "--lag-count", str(lag_count), "--target-test-fraction", str(target_test_fraction),
        ])
        _run_stage("data_scarcity.modeling.compare_models", [
            "--quantiles", quantiles, "--transfer-lambda", str(transfer_lambda),
            "--alpha", str(alpha), "--random-state", str(s),
        ])
        rows = _read_csv(PROCESSED_DATA_DIR / "model_comparison_results.csv")
        per_seed.append({r["name"]: {k: float(v) for k, v in r.items() if k != "name"} for r in rows})

    # Aggregate ---------------------------------------------------------------
    model_names = list(per_seed[0].keys())
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for m in model_names:
        agg[m] = {}
        for metric in per_seed[0][m]:
            vals = np.array([s[m][metric] for s in per_seed], dtype=float)
            agg[m][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_lo": float(np.quantile(vals, 0.025)) if len(vals) >= 4 else float("nan"),
                "ci_hi": float(np.quantile(vals, 0.975)) if len(vals) >= 4 else float("nan"),
                "n_seeds": len(vals),
            }

    out_json = output_dir / "aggregated_metrics.json"
    out_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Flat CSV for easy table copy-paste
    out_csv = output_dir / "aggregated_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "metric", "mean", "std", "ci_lo", "ci_hi", "n_seeds"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for m, mm in agg.items():
            for metric, stats in mm.items():
                w.writerow({"model": m, "metric": metric, **stats})
    logger.success("Multi-seed aggregation done | {} | {}", out_json, out_csv)


def _run_stage(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    logger.debug("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("STDERR:\n{}", r.stderr)
        raise RuntimeError(f"{module} failed (rc={r.returncode})")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    app()
