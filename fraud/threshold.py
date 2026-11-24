"""Pick the alarm threshold by money, not by the default 0.5.

Cost model (assumptions, change them to match a real business):
  * Missed fraud (false negative): we lose the transaction Amount.
  * False alarm (false positive): an analyst reviews it and the customer is
    annoyed -> fixed REVIEW_COST dollars.
  * Caught fraud (true positive): we still pay REVIEW_COST to review it.

The threshold is chosen on the validation set and then frozen, so the test
set result is an honest estimate.
"""
from __future__ import annotations

import numpy as np

REVIEW_COST = 10.0


def expected_cost(y_true, scores, amounts, threshold: float, review_cost: float = REVIEW_COST) -> float:
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=float)
    flag = np.asarray(scores) >= threshold
    missed_fraud = amounts[(y_true == 1) & ~flag].sum()
    reviews = flag.sum() * review_cost
    return float(missed_fraud + reviews)


def best_threshold(y_true, scores, amounts, review_cost: float = REVIEW_COST,
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    if grid is None:
        grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999),
                                         np.quantile(scores, np.linspace(0.9, 1, 201))]))
    costs = [expected_cost(y_true, scores, amounts, t, review_cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])
