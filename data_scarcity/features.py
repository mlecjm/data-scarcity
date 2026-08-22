"""Feature engineering for low-data demand forecasting.

Adds the calendar and rolling-mean features that the original manuscript
*claimed* but never produced.  Backwards-compatible with the previous
``lag_1..lag_K`` output: callers that only need lag features can pass
``--no-calendar --no-rolling``.

Outputs three CSVs:

* ``source_features.csv`` — features for the data-rich source pool.
* ``target_train_features.csv`` — first (1 - test_fraction) of each target series.
* ``target_test_features.csv`` — last test_fraction of each target series.

A leakage check is performed at write time: for every series the maximum
training timestamp must be strictly less than the minimum test timestamp.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Iterable

from loguru import logger
import typer

from data_scarcity.config import PROCESSED_DATA_DIR

app = typer.Typer(no_args_is_help=False, add_completion=False)


M5_REFERENCE_DATE = date(2011, 1, 29)  # M5 day_1 corresponds to 2011-01-29


@app.command()
def main(
    source_input_path: Path = PROCESSED_DATA_DIR / "source_dataset.csv",
    target_input_path: Path = PROCESSED_DATA_DIR / "target_dataset.csv",
    source_features_path: Path = PROCESSED_DATA_DIR / "source_features.csv",
    target_train_features_path: Path = PROCESSED_DATA_DIR / "target_train_features.csv",
    target_test_features_path: Path = PROCESSED_DATA_DIR / "target_test_features.csv",
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "y",
    lag_count: int = 7,
    target_test_fraction: float = 0.3,
    calendar: bool = True,
    rolling: bool = True,
) -> None:
    """Generate features and a leakage-free temporal split for the target pool."""
    if lag_count < 1:
        raise ValueError("lag_count must be >= 1")
    if not 0.0 < target_test_fraction < 1.0:
        raise ValueError("target_test_fraction must be in (0, 1)")

    source_rows = _read_rows(source_input_path)
    target_rows = _read_rows(target_input_path)

    source_features, fcols = _build_features(
        source_rows, series_col, time_col, target_col, lag_count, calendar, rolling
    )
    target_features, _ = _build_features(
        target_rows, series_col, time_col, target_col, lag_count, calendar, rolling
    )
    target_train, target_test = _split_target_temporal(
        target_features, series_col=series_col, time_col=time_col, test_fraction=target_test_fraction
    )
    _no_leakage_assert(target_train, target_test, series_col, time_col)

    fieldnames = [series_col, time_col, *fcols, "y"]
    _write_rows(source_features_path, fieldnames, source_features)
    _write_rows(target_train_features_path, fieldnames, target_train)
    _write_rows(target_test_features_path, fieldnames, target_test)

    logger.success(
        "Features generated | src={} tgt_train={} tgt_test={} | cols={}",
        len(source_features), len(target_train), len(target_test), len(fcols),
    )


# ---------------------------------------------------------------------------
def _build_features(
    rows: list[dict[str, str]],
    series_col: str,
    time_col: str,
    target_col: str,
    lag_count: int,
    calendar: bool,
    rolling: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[series_col]].append(row)

    feat_cols = [f"lag_{k}" for k in range(1, lag_count + 1)]
    rolling_windows = (7, 14) if rolling else ()
    feat_cols.extend(f"rmean_{w}" for w in rolling_windows)
    if calendar:
        feat_cols.extend(["dow_sin", "dow_cos", "moy_sin", "moy_cos"])

    out: list[dict[str, str]] = []
    for sid, srows in grouped.items():
        ordered = sorted(srows, key=lambda r: _time_sort_key(r[time_col]))
        y_vals = [float(r[target_col]) for r in ordered]
        n = len(ordered)

        # rolling means need >= max(window) past observations
        warmup = max(lag_count, *rolling_windows) if rolling_windows else lag_count
        for idx in range(warmup, n):
            item: dict[str, str] = {series_col: sid, time_col: ordered[idx][time_col], "y": str(y_vals[idx])}
            for k in range(1, lag_count + 1):
                item[f"lag_{k}"] = str(y_vals[idx - k])
            for w in rolling_windows:
                item[f"rmean_{w}"] = str(sum(y_vals[idx - w: idx]) / w)
            if calendar:
                ts = ordered[idx][time_col]
                dow, moy = _calendar_features(ts)
                item["dow_sin"] = str(math.sin(2 * math.pi * dow / 7))
                item["dow_cos"] = str(math.cos(2 * math.pi * dow / 7))
                item["moy_sin"] = str(math.sin(2 * math.pi * moy / 12))
                item["moy_cos"] = str(math.cos(2 * math.pi * moy / 12))
            out.append(item)

    return out, feat_cols


def _calendar_features(timestamp: str) -> tuple[int, int]:
    """Return (day_of_week 0-6, month 1-12).  Handles M5 d_K and ISO dates."""
    if timestamp.startswith("d_") and timestamp[2:].isdigit():
        offset_days = int(timestamp[2:]) - 1
        d = M5_REFERENCE_DATE + timedelta(days=offset_days)
        return d.weekday(), d.month
    try:
        from datetime import datetime
        d = datetime.fromisoformat(timestamp).date()
        return d.weekday(), d.month
    except ValueError:
        return 0, 1  # default fallback (constant column, ridge ignores)


def _split_target_temporal(
    rows: list[dict[str, str]], series_col: str, time_col: str, test_fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r[series_col]].append(r)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for sid, g in grouped.items():
        ordered = sorted(g, key=lambda r: _time_sort_key(r[time_col]))
        n = len(ordered)
        n_test = max(1, int(n * test_fraction))
        if n - n_test < 1:
            n_test = max(1, n - 1)
        split_idx = n - n_test
        train_rows.extend(ordered[:split_idx])
        test_rows.extend(ordered[split_idx:])
    return train_rows, test_rows


def _no_leakage_assert(
    train: list[dict[str, str]], test: list[dict[str, str]], series_col: str, time_col: str,
) -> None:
    tr: dict[str, float] = {}
    for r in train:
        k = _time_sort_key_float(r[time_col])
        tr[r[series_col]] = max(tr.get(r[series_col], float("-inf")), k)
    for r in test:
        if _time_sort_key_float(r[time_col]) <= tr.get(r[series_col], float("-inf")):
            raise AssertionError(
                f"Temporal leakage: test ts {r[time_col]} <= max train ts for series {r[series_col]}"
            )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _time_sort_key(value: str) -> float | str:
    if value.startswith("d_") and value[2:].isdigit():
        return float(int(value[2:]))
    try:
        return float(value)
    except ValueError:
        return value


def _time_sort_key_float(value: str) -> float:
    k = _time_sort_key(value)
    return float(k) if isinstance(k, float) else float("-inf")


if __name__ == "__main__":
    app()
