# Deployment & Execution Guide

*TR-MQLR: Transfer-Regularized Probabilistic Supply Chain Demand Forecasting Model Framework for Launch-Phase Retail under Data Scarcity* — AI2M4RI 2026.

**Repository release:** `v0.2.0` — https://github.com/mlecjm/data-scarcity/

This guide walks a reviewer (or any third party) through every step needed to install, configure, run, and troubleshoot the revised pipeline — from a clean machine to the full set of tables and figures of the manuscript. Three deployment options are documented:

1. **Local Python (recommended for development).**
2. **Docker / Docker Compose (recommended for one-command reproducibility).**
3. **Containerised CI / cluster execution (for batch sweeps).**

Pick the option that fits the environment. All three execute exactly the same code paths.

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Operating system | Linux, macOS, or Windows 10+ | The Makefile is cross-platform; no `cmd /c` commands. |
| Python | 3.11 | 3.12 also tested in CI. |
| RAM | 4 GB | Synthetic pipeline runs in well under 1 GB. |
| Disk | 500 MB | Source + features + outputs. |
| Docker | 24+ | Only required for Option 2. |
| GPU | none | Models are linear / sklearn — no accelerator needed. |

The M5 raw CSV is **optional** — `scripts/make_synthetic_m5.py` produces an offline drop-in replacement that exercises the same code paths.

---

## 2. Option 1 — Local Python

### 2.1 Clone & install

```bash
git clone https://github.com/mlecjm/data-scarcity.git
cd data-scarcity
git checkout v0.2.0          # pin to the revised release
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

For development tooling (pytest, ruff, black):

```bash
pip install -e ".[dev]"
```

### 2.2 Configure (optional)

Copy `.env.example` to `.env` and override any default. The most useful knobs:

```ini
DS_SEED=42                         # global reproducibility seed
# DS_DATA_DIR=/abs/path/to/data    # default: <repo>/data
# DS_MODELS_DIR=/abs/path/to/models
# DS_REPORTS_DIR=/abs/path/to/reports
```

These paths are read by `data_scarcity.config` at import time. The defaults work out of the box; override them only if you keep large datasets outside the repository tree.

### 2.3 Bring data into place

**Offline / synthetic:**

```bash
python scripts/make_synthetic_m5.py data/raw/m5/sales_train_validation.csv
```

This writes a 260-series × 400-day M5-shaped CSV in ~5 seconds. The generated file is structurally identical to `sales_train_validation.csv` (same `id`, `d_1`, …, `d_400` columns) so every downstream CLI sees no difference.

**Real M5:**

1. Register on Kaggle and download the [M5 Forecasting — Uncertainty](https://www.kaggle.com/competitions/m5-forecasting-uncertainty) competition data.
2. Unzip `sales_train_validation.csv` into `data/raw/m5/`.

### 2.4 Run the full pipeline

```bash
make reproduce
```

This is the *single* command that produces every result in the revised manuscript. Internally it runs:

1. `data_scarcity.dataset` — build source/target splits at the series level (204/46 split, seeded).
2. `data_scarcity.features` — generate lag + rolling-mean + sine/cosine calendar features; enforce per-series temporal leakage assertion.
3. `data_scarcity.modeling.compare_models` — fit 10 models (naïve, seasonal-naïve, EWMA, Croston, ridge target-only, ridge source-only, ridge transfer, random forest, gradient boosting, QR target-only, QR transfer) and emit the per-model metric panel.
4. `data_scarcity.modeling.run_experiment` — sweep the (history × λ) grid.
5. `data_scarcity.metrics_plots` — render all four figures.

On a 2024-class laptop the synthetic run takes ~3 minutes, the real M5 run ~6 minutes.

### 2.5 Run individual stages

Every stage is a `typer` CLI with `--help`:

```bash
python -m data_scarcity.dataset --help
python -m data_scarcity.features --help
python -m data_scarcity.modeling.compare_models --help
python -m data_scarcity.modeling.run_experiment --help
python -m data_scarcity.modeling.multi_seed_runner --help
python -m data_scarcity.metrics_plots --help
```

Examples:

```bash
# minimal: 100 series, 300 days, 5-day lag window
python -m data_scarcity.dataset --input-path data/raw/m5/sales_train_validation.csv \
       --max-series 100 --max-days 300 --target-history-fraction 0.2 --seed 7
python -m data_scarcity.features --lag-count 5

# compare models with a 90% predictive interval and a different transfer strength
python -m data_scarcity.modeling.compare_models --quantiles "0.05,0.5,0.95" \
       --alpha 0.05 --transfer-lambda 50

# 5-seed aggregation
python -m data_scarcity.modeling.multi_seed_runner --n-seeds 5 --transfer-lambda 10
```

### 2.6 Tests

```bash
pytest -q                       # 19 unit tests, ~3 s
pytest -q -m slow               # adds the end-to-end pipeline test, ~30 s
pytest -q --cov=data_scarcity   # with coverage
```

---

## 3. Option 2 — Docker / Docker Compose

This is the recommended path for reviewers who want a single-command, environment-independent reproduction.

### 3.1 Build the image

```bash
docker build -t data-scarcity:latest -f docker/Dockerfile .
```

The Dockerfile pins Python 3.11 and installs `requirements.txt`. Total image size ≈ 700 MB.

### 3.2 Run the pipeline

```bash
docker run --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/reports:/workspace/reports" \
  -v "$PWD/models:/workspace/models" \
  data-scarcity:latest make reproduce
```

Or with Docker Compose:

```bash
docker compose -f docker/docker-compose.yml run --rm reproduce
```

The volumes ensure the generated `data/processed/`, `reports/figures/`, and `models/` directories appear on the host.

### 3.3 Interactive shell inside the container

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  data-scarcity:latest bash
```

Inside the container you can run any individual CLI exactly as in Option 1.

---

## 4. Option 3 — Cluster / CI batch sweeps

For multi-seed or multi-grid sweeps the `multi_seed_runner` CLI is the entry point:

```bash
python -m data_scarcity.modeling.multi_seed_runner \
       --input-path data/raw/m5/sales_train_validation.csv \
       --history-fraction 0.2 \
       --transfer-lambda 10 \
       --n-seeds 5 \
       --output-dir data/processed/multi_seed
```

This writes:

* `data/processed/multi_seed/raw_runs.csv` — per-seed × per-model metrics.
* `data/processed/multi_seed/aggregated_metrics.csv` — mean ± std per model.
* `data/processed/multi_seed/bootstrap_ci.json` — paired-bootstrap CIs vs ridge target-only.

For a GitHub Actions-friendly workflow:

```yaml
- name: Reproduce
  run: |
    python scripts/make_synthetic_m5.py data/raw/m5/sales_train_validation.csv
    make reproduce
    pytest -q
- uses: actions/upload-artifact@v4
  with: { name: figures, path: reports/figures/ }
```

---

## 5. Output layout

After a successful run:

```
data/processed/
├── source_dataset.csv                     # 204 series source pool
├── target_dataset.csv                     # 46 series target pool (truncated)
├── source_features.csv
├── target_train_features.csv
├── target_test_features.csv
├── model_comparison_results.csv           # per-model metrics
├── model_comparison_predictions.csv       # per-row predictions + intervals
├── experiment_results_grid.csv            # history × lambda RMSE
├── multi_seed/
│   ├── raw_runs.csv
│   ├── aggregated_metrics.csv
│   └── bootstrap_ci.json
└── multi_seed_3.json                      # this submission's 3-seed validation

models/
└── experimental_model.json                # persisted weights + transfer config

reports/figures/
├── model_metrics_bars.png                 # 4-panel MAE/RMSE/pinball/Winkler
├── coverage_width_tradeoff.png            # uncertainty trade-off
├── transfer_gain_heatmap.png              # (history × lambda) heatmap
└── test_series_confidence_band.png        # observed + band, clipped at 0
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: data_scarcity` | `PYTHONPATH` not set | run from repo root or `export PYTHONPATH=$(pwd)`; the Makefile sets it automatically. |
| `FileNotFoundError: data/raw/m5/...` | M5 file missing | run `python scripts/make_synthetic_m5.py data/raw/m5/sales_train_validation.csv` for the offline fixture. |
| Empty `model_comparison_results.csv` | `dataset` produced no target rows | lower `--target-history-fraction` (some series have very short history) or increase `--max-days`. |
| Coverage stays at ~0.79 | The original (non-conformal) pipeline was used | make sure you're calling `compare_models`, not the legacy `train`/`predict` pair; the conformal step lives only in `compare_models`. |
| Figure shows band below 0 | Old figure cached | delete `reports/figures/test_series_confidence_band.png` and re-run `python -m data_scarcity.metrics_plots`. |
| `pytest` reports `PytestUnknownMarkWarning: slow` | `slow` mark not registered in your `pytest.ini` | harmless; the test is filtered out by default. To silence the warning, add `markers = ["slow"]` under `[tool.pytest.ini_options]` in `pyproject.toml`. |
| Docker run fails on Apple Silicon | image built for amd64 | rebuild with `docker build --platform linux/arm64 ...` or run with `--platform linux/amd64` if you accept emulation. |
| Subgradient optimisation oscillates | learning rate too high | lower `--lr` in `compare_models` (default 0.05); the test fixture uses 0.05 / 400 epochs and converges reliably. |

---

## 7. Verifying that the deployment is correct

Run the following smoke check after any installation:

```bash
python -c "
from data_scarcity import config
from data_scarcity.metrics import pinball_loss, paired_bootstrap_ci
from data_scarcity.modeling.quantile_regression import fit_quantile_linear, MultiQuantileLinear
from data_scarcity.modeling.conformal import fit_cqr_offset
from data_scarcity.decisions import NewsvendorConfig, newsvendor_orders
print('OK', config.PROJ_ROOT)
"
pytest -q -k "not slow"
```

The expected output ends with `19 passed`. If both the import and the test suite pass, the deployment is correct and `make reproduce` is guaranteed to terminate successfully.

---

## 8. Contact

For questions about reproduction or to report a bug, open an issue at
https://github.com/mlecjm/data-scarcity/issues. 
