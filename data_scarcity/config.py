"""Configuration and path management for the data_scarcity pipeline.

All paths are derived from PROJ_ROOT (= the repo root).  Override via env vars
(see ``.env.example``) when running inside Docker or a different layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


PROJ_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = _env_path("DS_DATA_DIR", PROJ_ROOT / "data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = _env_path("DS_MODELS_DIR", PROJ_ROOT / "models")
REPORTS_DIR: Path = _env_path("DS_REPORTS_DIR", PROJ_ROOT / "reports")
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Reproducibility ------------------------------------------------------------
GLOBAL_SEED: int = int(os.environ.get("DS_SEED", "42"))


@dataclass(frozen=True)
class FeatureConfig:
    """Static configuration for the feature engineering stage."""

    lag_count: int = 7
    calendar: bool = True            # day-of-week, month, week-of-year (sin/cos)
    rolling_means: tuple[int, ...] = (7, 14)
    include_promo: bool = False       # if event/promotion columns are present


DEFAULT_FEATURES = FeatureConfig()

# Logging --------------------------------------------------------------------
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
except ModuleNotFoundError:
    pass

logger.debug(f"PROJ_ROOT = {PROJ_ROOT}")
