import numpy as np

from fraud import data, leakage, models, threshold


def test_time_split_is_ordered_and_complete():
    df = data.make_synthetic(n=5000)
    s = data.time_split(df, "synthetic")
    assert len(s.y_train) + len(s.y_val) + len(s.y_test) == len(df)
    assert "Time" not in s.X_train.columns  # raw Time must not be a feature
    assert set(data.FEATURES) == set(s.X_train.columns)


def test_lazy_baseline_has_high_accuracy_and_zero_recall():
    y = np.array([0] * 998 + [1] * 2)
    r = models.always_legit_baseline(y)
    assert r["accuracy"] == 0.998 and r["recall"] == 0.0


def test_cost_flag_everything_vs_nothing():
    y = np.array([0, 0, 1, 1])
    amt = np.array([5.0, 5.0, 100.0, 300.0])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert threshold.expected_cost(y, scores, amt, 1.01, review_cost=10) == 400.0  # flag nothing
    assert threshold.expected_cost(y, scores, amt, 0.0, review_cost=10) == 40.0    # flag all 4
    assert threshold.expected_cost(y, scores, amt, 0.5, review_cost=10) == 20.0    # perfect


def test_smote_before_split_inflates_score():
    df = data.make_synthetic(n=8000, fraud_rate=0.01)
    r = leakage.smote_leakage_demo(df[data.FEATURES[:-1]], df["Class"], kind="logistic", folds=3)
    assert r["wrong_pr_auc_mean"] > r["right_pr_auc_mean"]
