PROJECT_NAME := data-scarcity
PYTHON_VERSION := 3.11
PYTHON ?= python
PYTHONPATH := $(CURDIR)
export PYTHONPATH

# raw → processed paths (overridable via env)
RAW_M5 ?= data/raw/m5/sales_train_validation.csv
HISTORY_FRACTION ?= 0.3
LAG_COUNT ?= 7
LAMBDA ?= 10
ALPHA ?= 0.1
QUANTILES ?= "0.1,0.5,0.9"
SEED ?= 42

.PHONY: help requirements clean lint format test \
        synthetic-data dataset features train predict compare_models \
        run_experiment plot metrics_plots pipeline reproduce multi_seed \
        docker-build docker-run

help:
	@echo "Targets:"
	@echo "  requirements        install Python dependencies"
	@echo "  synthetic-data      generate an offline M5-shaped CSV for validation"
	@echo "  dataset             build source/target splits"
	@echo "  features            create lag + calendar + rolling features"
	@echo "  train               train baseline + transfer + quantile head"
	@echo "  predict             evaluate on target test"
	@echo "  compare_models      compare 10 models w/ conformal calibration"
	@echo "  run_experiment      run history × lambda grid"
	@echo "  multi_seed          run pipeline across N seeds"
	@echo "  plot / metrics_plots  render manuscript figures"
	@echo "  pipeline            dataset → features → train → predict"
	@echo "  reproduce           full pipeline + grid + multi-seed + plots"
	@echo "  test                pytest"
	@echo "  lint / format       ruff + black"
	@echo "  docker-build        docker build -t data-scarcity:latest"
	@echo "  docker-run          docker run reproduce"

requirements:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

clean:
	$(PYTHON) -c "import pathlib, shutil; p=pathlib.Path('.'); [f.unlink(missing_ok=True) for f in p.rglob('*.pyc')]; [shutil.rmtree(d, ignore_errors=True) for d in p.rglob('__pycache__')]"

lint:
	ruff check .
	black --check .

format:
	ruff check . --fix
	black .

test:
	$(PYTHON) -m pytest -q -k "not slow"

# ---------------------------------------------------------------------------
synthetic-data:
	$(PYTHON) scripts/make_synthetic_m5.py $(RAW_M5)

dataset:
	$(PYTHON) -m data_scarcity.dataset --input-path $(RAW_M5) --max-series 250 --max-days 365 --target-history-fraction $(HISTORY_FRACTION) --seed $(SEED)

features:
	$(PYTHON) -m data_scarcity.features --lag-count $(LAG_COUNT) --target-test-fraction 0.3

train:
	$(PYTHON) -m data_scarcity.modeling.train --transfer-lambda $(LAMBDA) --ridge-lambda 0.001 --quantiles $(QUANTILES)

predict:
	$(PYTHON) -m data_scarcity.modeling.predict --features-path data/processed/target_test_features.csv --alpha $(ALPHA)

compare_models:
	$(PYTHON) -m data_scarcity.modeling.compare_models --quantiles $(QUANTILES) --ridge-alpha 0.001 --transfer-lambda $(LAMBDA) --alpha $(ALPHA) --random-state $(SEED)

run_experiment:
	$(PYTHON) -m data_scarcity.modeling.run_experiment --sales-input-path $(RAW_M5) --history-grid "0.1,0.2,0.3" --lambda-grid "1,10,50" --lag-count $(LAG_COUNT) --target-test-fraction 0.3 --quantiles $(QUANTILES) --alpha $(ALPHA) --max-series 250 --max-days 365

multi_seed:
	$(PYTHON) -m data_scarcity.modeling.multi_seed_runner --input-path $(RAW_M5) --history-fraction $(HISTORY_FRACTION) --transfer-lambda $(LAMBDA) --n-seeds 5

plot:
	$(PYTHON) -m data_scarcity.plots --features-path data/processed/target_test_features.csv --predictions-path data/processed/model_comparison_predictions.csv --output-path reports/figures/test_series_confidence_band.png --model-name ridge_transfer

metrics_plots:
	$(PYTHON) -m data_scarcity.metrics_plots

pipeline: dataset features train predict
	@echo "Pipeline complete."

reproduce: dataset features compare_models run_experiment metrics_plots
	@echo "Full reproduction pipeline complete."

# ---------------------------------------------------------------------------
docker-build:
	docker build -t $(PROJECT_NAME):latest -f docker/Dockerfile .

docker-run:
	docker compose -f docker/docker-compose.yml run --rm reproduce
