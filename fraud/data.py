"""Load the credit-card fraud data and split it by time.

The real dataset (Kaggle: mlg-ulb/creditcardfraud) has 284,807 card
transactions made over two days in September 2013, 492 of them fraud.
Columns: Time (seconds since the first transaction), V1..V28 (anonymised
PCA components), Amount, Class (1 = fraud).

If the CSV is missing, `make_synthetic` builds a small fake dataset with the
same columns so the code and tests can run. Results from synthetic data mean
nothing about real fraud and must never be reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
# Model features. Raw "Time" is left out on purpose: with a time-based split,
# test rows have Time values the model never saw, so trees would learn
# "early = this pattern" rules that can't transfer. Hour-of-day can.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
TARGET = "Class"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    return df


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    source: str  # "real" or "synthetic"


def make_synthetic(n: int = 40_000, fraud_rate: float = 0.002, seed: int = 0) -> pd.DataFrame:
    """Fake data with the real schema. For smoke tests only."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    V = rng.normal(size=(n, 28))
    # Make a handful of components informative, like V14/V17/V12 in the real data.
    for j, shift in [(13, -4.0), (16, -3.5), (11, -3.0), (9, -2.0), (3, 2.0)]:
        V[y == 1, j] += shift + rng.normal(scale=1.5, size=y.sum())
    amount = np.round(rng.lognormal(3.0, 1.3, size=n), 2)
    amount[y == 1] = np.round(rng.lognormal(3.5, 1.6, size=y.sum()), 2)
    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, size=n)))
    df["Amount"] = amount
    df[TARGET] = y
    return df


def load(csv_path: str | Path = "data/creditcard.csv") -> tuple[pd.DataFrame, str]:
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        missing = set(RAW_COLUMNS + [TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return df, "real"
    print(f"[data] {path} not found -> using SYNTHETIC data (numbers are not meaningful)")
    return make_synthetic(), "synthetic"


def time_split(df: pd.DataFrame, source: str, train: float = 0.6, val: float = 0.2) -> Splits:
    """Oldest 60% -> train, next 20% -> validation, newest 20% -> test.

    Splitting by time (not at random) mimics production: the model is always
    judged on transactions that happen *after* the ones it learned from.
    """
    df = add_features(df).sort_values("Time").reset_index(drop=True)
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    parts = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    out = []
    for p in parts:
        out += [p[FEATURES].reset_index(drop=True), p[TARGET].reset_index(drop=True)]
    return Splits(*out, source=source)
