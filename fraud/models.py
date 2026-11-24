"""Supervised models and the metrics we judge them by."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


def imbalance_ratio(y: pd.Series) -> float:
    """negatives / positives, e.g. ~580 for the real data."""
    pos = int(y.sum())
    return (len(y) - pos) / max(pos, 1)


def build_models(y_train: pd.Series) -> dict:
    ratio = imbalance_ratio(y_train)
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            # Tell the model a missed fraud matters ~ratio times more.
            scale_pos_weight=ratio,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def score_report(y_true, scores, threshold: float = 0.5) -> dict:
    """PR-AUC first: it's the metric that stays honest when fraud is rare."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "flagged": int(pred.sum()),
    }


def always_legit_baseline(y_true) -> dict:
    """The 'lazy model' that says nobody is a fraudster. Huge accuracy, useless."""
    y_true = np.asarray(y_true)
    return {
        "accuracy": accuracy_score(y_true, np.zeros_like(y_true)),
        "recall": 0.0,
        "pr_auc": float(y_true.mean()),  # PR-AUC of a random/constant scorer = fraud rate
    }
