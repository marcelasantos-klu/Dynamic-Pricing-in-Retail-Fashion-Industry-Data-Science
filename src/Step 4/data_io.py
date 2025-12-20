"""
Data loading and splitting utilities.

Responsibilities:
- load_data(): read the prepared CSV, normalize column names.
- make_train_test_split(): apply configured split (city stratify where requested) with a safe fallback to non-stratified.
- Tee: simple stdout duplicator so logs are captured and still printed.
"""
from __future__ import annotations

import sys
import io
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, DEFAULT_SPLIT_STRATEGY, SPLIT_STRATEGIES


def load_data() -> pd.DataFrame:
    """Load the prepared Airbnb dataset from disk and normalize column names."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Crime Index": "Crime_Index"})
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


def make_train_test_split(
    X: pd.DataFrame,
    y_log: pd.Series,
    city_series: pd.Series,
    strategy_key: Optional[str] = None,
):
    """Create a train/test split according to a named strategy."""
    key = strategy_key or DEFAULT_SPLIT_STRATEGY
    cfg = SPLIT_STRATEGIES.get(key)
    if cfg is None:
        print(f"Unknown split '{key}', falling back to {DEFAULT_SPLIT_STRATEGY}")
        cfg = SPLIT_STRATEGIES[DEFAULT_SPLIT_STRATEGY]
        key = DEFAULT_SPLIT_STRATEGY
    stratify = city_series if cfg.get("stratify_by_city") else None
    try:
        return train_test_split(
            X, y_log,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            stratify=stratify,
        )
    except ValueError as e:
        print(f"Warning: stratified split failed ({e}). Falling back to non-stratified split.")
        return train_test_split(
            X, y_log,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            stratify=None,
        )


class Tee(io.StringIO):  # type: ignore[name-defined]
    """Simple tee: write to stdout and buffer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        return super().write(s)

    def flush(self):
        self._stdout.flush()
        return super().flush()
