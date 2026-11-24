"""Reason codes: the top features that pushed each flagged transaction toward fraud.

Uses SHAP TreeExplainer on the XGBoost model. A positive SHAP value means the
feature pushed the score toward fraud for *this* transaction.

Caveat: V1..V28 in the public dataset are anonymised PCA components, so a
reason code reads "V14 = -7.2" rather than "card used in a new country".
In a real bank the features have business names, and that's where reason
codes become useful to investigators and regulators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    return shap.TreeExplainer(model).shap_values(X)


def reason_codes(model, X: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    sv = shap_values(model, X)
    rows = []
    for i in range(len(X)):
        order = np.argsort(-sv[i])[:top_k]
        row = {"row": X.index[i]}
        for r, j in enumerate(order, start=1):
            feat = X.columns[j]
            row[f"reason_{r}"] = f"{feat}={X.iloc[i, j]:.2f} ({sv[i, j]:+.2f})"
        rows.append(row)
    return pd.DataFrame(rows)
