"""Run the whole project end to end and write reports/results.md.

    python run_all.py                      # uses data/creditcard.csv if present
    python run_all.py --csv path/to.csv
    python run_all.py --quick              # smaller settings, for CI
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud import anomaly, data, explain, leakage, models, threshold

OUT = Path("reports")
FIG = OUT / "figures"


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main(csv: str, quick: bool) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, source = data.load(csv)
    s = data.time_split(df, source)
    F = data.FEATURES
    res: dict = {"source": source, "rows": len(df), "fraud_rows": int(df[data.TARGET].sum()),
                 "fraud_rate": float(df[data.TARGET].mean()),
                 "split_sizes": {"train": len(s.y_train), "val": len(s.y_val), "test": len(s.y_test)},
                 "split_frauds": {"train": int(s.y_train.sum()), "val": int(s.y_val.sum()),
                                  "test": int(s.y_test.sum())}}
    print(f"[1/6] data: {source}, {len(df):,} rows, fraud rate {res['fraud_rate']:.4%}")

    # 1. The accuracy trap
    res["lazy_baseline_test"] = models.always_legit_baseline(s.y_test)

    # 2. Supervised models: train on train, compare on validation, report on test
    fitted, val_scores, test_scores = {}, {}, {}
    res["supervised"] = {}
    for name, m in models.build_models(s.y_train).items():
        m.fit(s.X_train[F], s.y_train)
        fitted[name] = m
        val_scores[name] = m.predict_proba(s.X_val[F])[:, 1]
        test_scores[name] = m.predict_proba(s.X_test[F])[:, 1]
        res["supervised"][name] = {"val": models.score_report(s.y_val, val_scores[name]),
                                   "test": models.score_report(s.y_test, test_scores[name])}
        print(f"[2/6] {name}: val PR-AUC {res['supervised'][name]['val']['pr_auc']:.4f}")
    best = max(res["supervised"], key=lambda k: res["supervised"][k]["val"]["pr_auc"])
    res["best_model_by_val_pr_auc"] = best

    # 3. Cost-based threshold for the best model (chosen on val, applied to test)
    t_best, _ = threshold.best_threshold(s.y_val, val_scores[best], s.X_val["Amount"])
    res["threshold"] = {
        "model": best,
        "review_cost_usd": threshold.REVIEW_COST,
        "chosen_threshold": t_best,
        "test_cost_at_0.5": threshold.expected_cost(s.y_test, test_scores[best], s.X_test["Amount"], 0.5),
        "test_cost_at_chosen": threshold.expected_cost(s.y_test, test_scores[best], s.X_test["Amount"], t_best),
        "test_cost_flag_nothing": float(s.X_test["Amount"][s.y_test == 1].sum()),
        "test_report_at_0.5": models.score_report(s.y_test, test_scores[best], 0.5),
        "test_report_at_chosen": models.score_report(s.y_test, test_scores[best], t_best),
    }
    print(f"[3/6] threshold {t_best:.3f} for {best}")

    # 4. Unsupervised detectors (never see a fraud label in training)
    Xtr, Xte = s.X_train[F], s.X_test[F]
    iso = anomaly.isolation_forest_scores(Xtr, s.y_train, Xte)
    ae = anomaly.autoencoder_scores(Xtr, s.y_train, Xte, epochs=5 if quick else 15)
    res["unsupervised_test"] = {
        "isolation_forest": {"pr_auc": average_precision_score(s.y_test, iso)},
        "autoencoder": {"pr_auc": average_precision_score(s.y_test, ae)},
    }
    print("[4/6] anomaly detectors done")

    # 5. Leakage demo on the training period only
    n = 30_000 if quick else len(s.X_train)
    idx = s.X_train.sample(n=min(n, len(s.X_train)), random_state=0).index
    res["smote_leakage"] = leakage.smote_leakage_demo(
        s.X_train.loc[idx, F], s.y_train.loc[idx], kind="xgboost", folds=3 if quick else 5)
    print(f"[5/6] leakage demo: {res['smote_leakage']}")

    # 6. Reason codes for the top-scored test transactions (XGBoost)
    xgb = fitted["xgboost"]
    top = np.argsort(-test_scores["xgboost"])[:10]
    Xtop = s.X_test[F].iloc[top]
    rc = explain.reason_codes(xgb, Xtop)
    rc.insert(1, "fraud_score", test_scores["xgboost"][top].round(4))
    rc.insert(2, "actually_fraud", s.y_test.iloc[top].values)
    rc.to_csv(OUT / "reason_codes_top10.csv", index=False)
    res["reason_codes_file"] = "reports/reason_codes_top10.csv"
    print("[6/6] reason codes written")

    # Figures
    plt.figure(figsize=(6, 5))
    for name, sc in test_scores.items():
        p, r, _ = precision_recall_curve(s.y_test, sc)
        plt.plot(r, p, label=f"{name} (AP={average_precision_score(s.y_test, sc):.3f})")
    for name, sc in [("isolation_forest", iso), ("autoencoder", ae)]:
        p, r, _ = precision_recall_curve(s.y_test, sc)
        plt.plot(r, p, "--", label=f"{name} (AP={average_precision_score(s.y_test, sc):.3f})")
    plt.axhline(s.y_test.mean(), color="grey", lw=0.8, label="random guessing")
    plt.xlabel("Recall (share of fraud caught)")
    plt.ylabel("Precision (share of alarms that were fraud)")
    plt.title(f"Precision-recall on test set ({source} data)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "pr_curves.png", dpi=130)
    plt.close()

    grid = np.linspace(0.01, 0.99, 99)
    costs = [threshold.expected_cost(s.y_val, val_scores[best], s.X_val["Amount"], t) for t in grid]
    plt.figure(figsize=(6, 4))
    plt.plot(grid, costs)
    plt.axvline(t_best, ls="--", color="red", label=f"chosen = {t_best:.3f}")
    plt.axvline(0.5, ls=":", color="grey", label="default 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Cost on validation set ($)")
    plt.title(f"Cost vs threshold ({best})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "cost_vs_threshold.png", dpi=130)
    plt.close()

    sample = s.X_test[F].sample(n=min(2000, len(s.X_test)), random_state=0)
    sv = explain.shap_values(xgb, sample)
    shap.summary_plot(sv, sample, show=False, max_display=12)
    plt.title("Which features push scores toward fraud (SHAP)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=130)
    plt.close()

    res["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, rc)
    return res


def write_markdown(r: dict, rc: pd.DataFrame) -> None:
    warn = "" if r["source"] == "real" else (
        "> **WARNING: these numbers come from SYNTHETIC data and say nothing about real fraud.**\n"
        "> Put `creditcard.csv` in `data/` and re-run.\n\n")
    L = [f"# Results ({r['source']} data)\n", warn,
         f"Rows: {r['rows']:,} | fraud rows: {r['fraud_rows']:,} | fraud rate: {r['fraud_rate']:.4%}  ",
         f"Time split (oldest→newest): train {r['split_sizes']['train']:,} "
         f"({r['split_frauds']['train']} fraud), val {r['split_sizes']['val']:,} "
         f"({r['split_frauds']['val']} fraud), test {r['split_sizes']['test']:,} "
         f"({r['split_frauds']['test']} fraud)\n",
         "## 1. The accuracy trap\n",
         f"A model that labels **every** transaction legit scores **{r['lazy_baseline_test']['accuracy']:.4%} "
         f"accuracy** on the test set and catches **0** fraud. Its PR-AUC is "
         f"{r['lazy_baseline_test']['pr_auc']:.4f} (= the fraud rate).\n",
         "## 2. Supervised models (test set, threshold 0.5)\n",
         "| model | PR-AUC | ROC-AUC | precision | recall | accuracy | alarms |",
         "|---|---|---|---|---|---|---|"]
    for name, d in r["supervised"].items():
        t = d["test"]
        L.append(f"| {name} | {t['pr_auc']:.4f} | {t['roc_auc']:.4f} | {t['precision']:.3f} | "
                 f"{t['recall']:.3f} | {t['accuracy']:.4f} | {t['flagged']} |")
    th = r["threshold"]
    L += [f"\nBest on validation PR-AUC: **{r['best_model_by_val_pr_auc']}**.\n",
          "## 3. Choosing the threshold by cost\n",
          f"Assumptions: missed fraud costs its Amount; every alarm costs ${th['review_cost_usd']:.0f} to review.\n",
          "| policy (test set) | cost | precision | recall | alarms |", "|---|---|---|---|---|",
          f"| flag nothing | ${th['test_cost_flag_nothing']:,.0f} | – | 0 | 0 |",
          f"| threshold 0.5 | ${th['test_cost_at_0.5']:,.0f} | {th['test_report_at_0.5']['precision']:.3f} | "
          f"{th['test_report_at_0.5']['recall']:.3f} | {th['test_report_at_0.5']['flagged']} |",
          f"| threshold {th['chosen_threshold']:.3f} (picked on val) | ${th['test_cost_at_chosen']:,.0f} | "
          f"{th['test_report_at_chosen']['precision']:.3f} | {th['test_report_at_chosen']['recall']:.3f} | "
          f"{th['test_report_at_chosen']['flagged']} |",
          "\n![cost](figures/cost_vs_threshold.png)\n",
          "## 4. Unsupervised detectors (no fraud labels used in training)\n",
          f"Isolation Forest PR-AUC: {r['unsupervised_test']['isolation_forest']['pr_auc']:.4f}  ",
          f"Autoencoder PR-AUC: {r['unsupervised_test']['autoencoder']['pr_auc']:.4f}\n",
          "![pr](figures/pr_curves.png)\n",
          "## 5. SMOTE leakage demo (cross-validated PR-AUC, training period)\n",
          "| how SMOTE was applied | PR-AUC (mean ± std) |", "|---|---|",
          f"| WRONG: before the split | {r['smote_leakage']['wrong_pr_auc_mean']:.4f} ± {r['smote_leakage']['wrong_pr_auc_std']:.4f} |",
          f"| RIGHT: inside each training fold | {r['smote_leakage']['right_pr_auc_mean']:.4f} ± {r['smote_leakage']['right_pr_auc_std']:.4f} |",
          "\nThe 'wrong' number is also scored on a test fold that contains fake SMOTE fraud rows, "
          "so it is not measuring real-world performance at all.\n",
          "## 6. Reason codes (top 10 highest-scored test transactions, XGBoost)\n",
          rc.to_markdown(index=False), "\n", "![shap](figures/shap_summary.png)\n",
          f"\n_Runtime: {r['runtime_seconds']} s_\n"]
    (OUT / "results.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.csv, a.quick)
