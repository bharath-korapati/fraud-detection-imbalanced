"""SMOTE done wrong vs SMOTE done right.

SMOTE invents new fraud rows by drawing points *between* real fraud rows.
If you run it on the whole dataset before cross-validation, the invented
rows in the training folds are near-copies of real fraud rows sitting in the
test fold. The model has effectively seen the test answers, so the score
goes up for a reason that won't exist in production.

Right way: put SMOTE inside an imblearn Pipeline, so it only ever runs on
the training part of each fold.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42


def _model(kind: str):
    if kind == "xgboost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             tree_method="hist", random_state=SEED, n_jobs=-1)
    return LogisticRegression(max_iter=2000)


def smote_leakage_demo(X, y, kind: str = "xgboost", folds: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # WRONG: oversample everything first, then cross-validate.
    scaler = StandardScaler().fit(X)  # also a (smaller) leak: scaler saw all rows
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(scaler.transform(X), y)
    wrong = cross_val_score(_model(kind), X_res, y_res, cv=cv, scoring="average_precision")

    # RIGHT: scaler + SMOTE are fit inside each training fold only.
    pipe = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=SEED)),
        ("model", _model(kind)),
    ])
    right = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")

    return {
        "wrong_pr_auc_mean": float(wrong.mean()),
        "wrong_pr_auc_std": float(wrong.std()),
        "right_pr_auc_mean": float(right.mean()),
        "right_pr_auc_std": float(right.std()),
    }
