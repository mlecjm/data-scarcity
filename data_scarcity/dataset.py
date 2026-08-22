"""Build source / target datasets for low-data forecasting experiments.

Accepts either the M5 wide format (``id``, ``d_1``, ..., ``d_N``) or a long
format (``series_id``, ``timestamp``, ``y``).  The target pool is sampled at
the *series* level (so a series belongs entirely to either source or target)
and target series are truncated to a fraction of their history to emulate the
launch-phase data-scarcity regime described in the manuscript.

This module is a streaming, memory-bounded re-write of the original
``dataset.py``.  It keeps the same CLI surface so existing scripts and the
``Makefile`` continue to work.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Iterable

from loguru import logger
import typer

from data_scarcity.config import GLOBAL_SEED, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(no_args_is_help=False, add_completion=False)


# ---------------------------------------------------------------------------
# Public CLI
# ---------------------------------------------------------------------------
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "m5" / "sales_train_validation.csv",
    source_output_path: Path = PROCESSED_DATA_DIR / "source_dataset.csv",
    target_output_path: Path = PROCESSED_DATA_DIR / "target_dataset.csv",
    metadata_output_path: Path = PROCESSED_DATA_DIR / "experimental_setup_metadata.json",
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "y",
    target_ratio: float = 0.2,
    target_history_fraction: float = 0.25,
    min_points_per_series: int = 20,
    max_series: int = 250,
    max_days: int = 365,
    seed: int = GLOBAL_SEED,
) -> None:
    """Build source/target CSVs and persist setup metadata for reproducibility."""
    _validate_args(target_ratio, target_history_fraction)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    logger.info("Loading raw dataset from {}", input_path)
    header = _peek_header(input_path)

    if "id" in header and any(name.startswith("d_") for name in header):
        _process_m5_stream(
            input_path=input_path,
            source_output_path=source_output_path,
            target_output_path=target_output_path,
            metadata_output_path=metadata_output_path,
            target_ratio=target_ratio,
            target_history_fraction=target_history_fraction,
            min_points_per_series=min_points_per_series,
            max_series=max_series,
            max_days=max_days,
            seed=seed,
        )
        return

    _process_long_format(
        input_path=input_path,
        source_output_path=source_output_path,
        target_output_path=target_output_path,
        metadata_output_path=metadata_output_path,
        series_col=series_col,
        time_col=time_col,
        target_col=target_col,
        target_ratio=target_ratio,
        target_history_fraction=target_history_fraction,
        min_points_per_series=min_points_per_series,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Format-specific processing
# ---------------------------------------------------------------------------
def _process_m5_stream(
    input_path: Path,
    source_output_path: Path,
    target_output_path: Path,
    metadata_output_path: Path,
    target_ratio: float,
    target_history_fraction: float,
    min_points_per_series: int,
    max_series: int,
    max_days: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    _ensure_parents(source_output_path, target_output_path, metadata_output_path)
    fieldnames = ["series_id", "timestamp", "y"]
    truncation_info: dict[str, dict[str, int]] = {}

    n_source_series = n_target_series = 0
    n_source_rows = n_target_rows = 0
    processed = 0

    with (
        input_path.open("r", newline="", encoding="utf-8") as fin,
        source_output_path.open("w", newline="", encoding="utf-8") as fs,
        target_output_path.open("w", newline="", encoding="utf-8") as ft,
    ):
        reader = csv.DictReader(fin)
        day_cols = sorted(
            [c for c in (reader.fieldnames or []) if c.startswith("d_")],
            key=_time_sort_key,
        )
        if max_days > 0:
            day_cols = day_cols[-max_days:]

        sw = csv.DictWriter(fs, fieldnames=fieldnames); sw.writeheader()
        tw = csv.DictWriter(ft, fieldnames=fieldnames); tw.writeheader()

        for row in reader:
            if 0 < max_series <= processed:
                break
            pairs = [(d, row[d]) for d in day_cols]
            if len(pairs) < min_points_per_series:
                continue
            processed += 1
            sid = row["id"]
            is_target = rng.random() < target_ratio

            if is_target:
                kept = max(8, int(len(pairs) * target_history_fraction))
                kept = min(kept, len(pairs))
                for d, y in pairs[:kept]:
                    tw.writerow({"series_id": sid, "timestamp": d, "y": y})
                    n_target_rows += 1
                n_target_series += 1
                truncation_info[sid] = {"original_points": len(pairs), "kept_points": kept}
            else:
                for d, y in pairs:
                    sw.writerow({"series_id": sid, "timestamp": d, "y": y})
                    n_source_rows += 1
                n_source_series += 1

    if n_source_series == 0 or n_target_series == 0:
        raise ValueError(
            "M5 split failed (empty source or target). Increase max_series or adjust target_ratio."
        )

    metadata = {
        "input_path": str(input_path),
        "source_output_path": str(source_output_path),
        "target_output_path": str(target_output_path),
        "n_series_total": n_source_series + n_target_series,
        "n_series_source": n_source_series,
        "n_series_target": n_target_series,
        "n_rows_source": n_source_rows,
        "n_rows_target": n_target_rows,
        "target_ratio": target_ratio,
        "target_history_fraction": target_history_fraction,
        "max_series": max_series,
        "max_days": max_days,
        "seed": seed,
        "truncation_per_target_series": truncation_info,
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.success(
        "M5 setup complete | source {}/{}r | target {}/{}r | seed={}",
        n_source_series, n_source_rows, n_target_series, n_target_rows, seed,
    )


def _process_long_format(
    input_path: Path,
    source_output_path: Path,
    target_output_path: Path,
    metadata_output_path: Path,
    series_col: str,
    time_col: str,
    target_col: str,
    target_ratio: float,
    target_history_fraction: float,
    min_points_per_series: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    missing = {series_col, time_col, target_col}.difference(fieldnames)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r[series_col]].append(r)
    valid = {
        sid: sorted(g, key=lambda r: _time_sort_key(r[time_col]))
        for sid, g in grouped.items()
        if len(g) >= min_points_per_series
    }
    if len(valid) < 2:
        raise ValueError("Need at least 2 valid series after filtering")

    series_ids = sorted(valid)
    n_target = max(1, min(len(series_ids) - 1, int(len(series_ids) * target_ratio)))
    target_ids = set(rng.sample(series_ids, k=n_target))

    source_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    truncation_info: dict[str, dict[str, int]] = {}

    for sid in sorted(valid):
        if sid in target_ids:
            full = valid[sid]
            kept = min(len(full), max(8, int(len(full) * target_history_fraction)))
            target_rows.extend(full[:kept])
            truncation_info[sid] = {"original_points": len(full), "kept_points": kept}
        else:
            source_rows.extend(valid[sid])

    _ensure_parents(source_output_path, target_output_path, metadata_output_path)
    _write_dicts(source_output_path, fieldnames, source_rows)
    _write_dicts(target_output_path, fieldnames, target_rows)

    metadata = {
        "input_path": str(input_path),
        "n_series_total": len(series_ids),
        "n_series_source": len(series_ids) - len(target_ids),
        "n_series_target": len(target_ids),
        "target_ratio": target_ratio,
        "target_history_fraction": target_history_fraction,
        "truncation_per_target_series": truncation_info,
        "seed": seed,
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.success(
        "Long-format setup complete | source rows {} | target rows {}",
        len(source_rows), len(target_rows),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_args(target_ratio: float, target_history_fraction: float) -> None:
    if not 0.0 < target_ratio < 1.0:
        raise ValueError("target_ratio must be in (0, 1)")
    if not 0.0 < target_history_fraction <= 1.0:
        raise ValueError("target_history_fraction must be in (0, 1]")


def _peek_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _ensure_parents(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def _write_dicts(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _time_sort_key(value: str) -> float | str:
    if value.startswith("d_"):
        suffix = value[2:]
        if suffix.isdigit():
            return float(int(suffix))
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    app()
