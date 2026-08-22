# TR-MQLR: Transfer-Regularized Probabilistic Supply Chain Demand Forecasting Model Framework for Launch-Phase Retail under Data Scarcity

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Companion repository for the paper  "*TR-MQLR: Transfer-Regularized Probabilistic Supply Chain Demand Forecasting Model Frameworkfor Launch-Phase Retail under Data Scarcity"* presented at the **I2M4RI 2026 paper**.

> **What's new in v0.2.0 (response-to-reviewers release)**
>
> * Honest, end-to-end re-implementation of the transfer mechanism as a *Gaussian-MAP / proximal-regularised linear model*, in line with the actual math — no more deep-learning iconography.
> * **True per-quantile linear regression** trained with the pinball loss (replaces the original point-plus-residual-shift trick).
> * **Split-conformal (CQR) calibration** of predictive intervals — fixes the systematic under-coverage observed in v0.1.
> * Calendar (day-of-week, month) and rolling-mean features (previously claimed but never produced).
> * Strong probabilistic and classical baselines: seasonal-naïve(7), exponentially-weighted lag average, Croston for intermittent demand, source-only ridge.
> * **Multi-seed runner** + paired bootstrap CIs for statistically meaningful comparisons.
> * **Inventory decision layer** (newsvendor with under/over-stock costs) so the paper's "Stage 4" promise is actually delivered.
> * Complete metric panel: MAE, RMSE, sMAPE, MASE, pinball, CRPS, coverage, width, Winkler interval score, latency.
> * **Dockerfile + cross-platform Makefile** for one-command reproducibility.

---

## Pipeline at a glance

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Data-Scarcity Pipeline                       │
│                                                                              │
│ raw M5 CSV ──► data_scarcity.dataset ──► source.csv  + target.csv (truncated)│
│                                                                              │
│ source/target ─► data_scarcity.features                                      │
│                  └─ lags(1..K) + rmean(7,14) + dow/moy sin/cos               │
│                                                                              │
│ src_features ─► data_scarcity.modeling.quantile_regression                   │
│                  └─ multi-quantile linear w/ source prior anchor             │
│                                                                              │
│ tgt_features ─► data_scarcity.modeling.compare_models                        │
│                  └─ 10 models + split-conformal calibration                  │
│                                                                              │
│ predictions ──► data_scarcity.decisions  (newsvendor → orders, cost, SL)     │
│                                                                              │
│ predictions ──► data_scarcity.metrics_plots                                  │
│                  └─ metric bars, coverage/width, predictive band, heatmap    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

### Option A — local Python (≥ 3.11)

```bash
git clone https://github.com/mlecjm/data-scarcity.git
cd data-scarcity
pip install -r requirements.txt

# offline-friendly synthetic dataset, lets the full pipeline run without M5
python scripts/make_synthetic_m5.py data/raw/m5/sales_train_validation.csv

make reproduce
```

`make reproduce` runs: dataset → features → 10-model comparison → 3×3 grid → all figures.
Outputs land in `data/processed/` and `reports/figures/`.

### Option B — Docker

```bash
make docker-build
make docker-run        # equivalent to: docker compose run --rm reproduce
```

### Option C — real M5 data

Download `sales_train_validation.csv` from the [M5 Forecasting — Uncertainty](https://www.kaggle.com/competitions/m5-forecasting-uncertainty) competition, drop it under `data/raw/m5/`, then run `make reproduce`.

## CLI reference

Every stage is a `typer` CLI that prints `--help`:

| Module                                                    | Purpose                                                        |
| --------------------------------------------------------- | -------------------------------------------------------------- |
| `data_scarcity.dataset`                                 | source/target split with truncation + reproducibility metadata |
| `data_scarcity.features`                                | lag + calendar + rolling-mean features (leakage-checked)       |
| `data_scarcity.modeling.train`                          | original point-only training (v0.1, kept for reproducibility)  |
| `data_scarcity.modeling.predict`                        | inference with v0.1 model                                      |
| `data_scarcity.modeling.compare_models`                 | **revised** harness: 10 models, full metric panel, CQR   |
| `data_scarcity.modeling.run_experiment`                 | (history × lambda) grid sweep                                 |
| `data_scarcity.modeling.multi_seed_runner`              | N-seed aggregation with bootstrap CIs                          |
| `data_scarcity.plots` / `data_scarcity.metrics_plots` | publication figures                                            |

## Project layout

```
data_scarcity/
├── __init__.py
├── config.py                       # paths + seed + FeatureConfig
├── dataset.py                      # source/target builder
├── features.py                     # lag + calendar + rolling
├── metrics.py                      # MAE/RMSE/sMAPE/MASE/pinball/CRPS/Winkler/bootstrap
├── decisions.py                    # newsvendor reorder + service level + cost
├── plots.py                        # per-series predictive band
├── metrics_plots.py                # bars / heatmap / coverage-width / band
└── modeling/
    ├── baselines.py                # naive, seasonal-naive(7), EWMA, Croston
    ├── conformal.py                # split-CQR offset
    ├── quantile_regression.py      # multi-quantile linear (subgradient pinball)
    ├── compare_models.py           # main evaluation harness
    ├── multi_seed_runner.py        # aggregator
    ├── train.py / predict.py       # original v0.1 single-model CLI
    └── run_experiment.py           # (history × lambda) grid

scripts/make_synthetic_m5.py        # offline synthetic generator
docker/                             # Dockerfile + compose
tests/                              # 19 unit tests + 1 slow end-to-end
```

## Outputs

| File                                                               | Content                                   |
| ------------------------------------------------------------------ | ----------------------------------------- |
| `data/processed/source_dataset.csv` / `target_dataset.csv`     | raw split                                 |
| `data/processed/source_features.csv` / `target_*_features.csv` | engineered features                       |
| `data/processed/model_comparison_results.csv`                    | per-model metrics                         |
| `data/processed/model_comparison_predictions.csv`                | per-row predictions + intervals           |
| `data/processed/experiment_results_grid.csv`                     | (history × lambda) RMSE table            |
| `data/processed/multi_seed/aggregated_metrics.{csv,json}`        | mean ± std + CIs                         |
| `reports/figures/model_metrics_bars.png`                         | 4-panel metric comparison                 |
| `reports/figures/transfer_gain_heatmap.png`                      | RMSE gain of ridge transfer               |
| `reports/figures/coverage_width_tradeoff.png`                    | uncertainty trade-off                     |
| `reports/figures/test_series_confidence_band.png`                | observed + predictive band (clipped at 0) |

## Citing

```bibtex
@inproceedings{datascarcity2026,
  title     = {Transfer-Regularised Probabilistic Demand Forecasting for Launch-Phase Retail under Data Scarcity},
  author    = {Merlec, M. M., Jean-Luc, M. K., Ulysse, M. K., Lucianne, K. K., and Patrick, K. T. },
  booktitle = {International Workshop on AI \& Mathematical Methods for Real-World Impact (AI2M4RI)},
  address   = {Athens, Greece},
  year      = {2026},
  publisher = {Elsevier Procedia Computer Science}
}
```

## License

MIT — see [LICENSE](LICENSE).
