"""
evaluate.py — Post-training evaluation gate for the fraud detection model.

Optional — NOT part of the mandatory etl.py -> feature_engineering.py ->
train.py chain. Nothing downstream needs this to run automatically; its
job is different: after a fresh train.py run, this script decides
whether the new model is good enough to deploy, and finds the best
classification threshold for it (Q1-4_evaluation.ipynb Cells 35-36).

Run it manually after training, or wire it in as a CI check that has
to pass before a model is allowed to be promoted/deployed — that's the
"quality gate" pattern real MLOps pipelines use.

Usage:
    python src/evaluate.py --data data/processed/df_final.csv --model models/model.pkl --out models/
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Minimum bar a model must clear to be considered deployable.
# This is a real "quality gate" — tune these numbers to whatever your
# project actually requires.
MIN_PR_AUC = 0.45
MIN_RECALL_AT_DEFAULT_THRESHOLD = 0.60


def load_test_split(data_path: str):
    """Reconstruct the exact same test split train.py used (same
    random_state=42 / stratify=y), so evaluation runs on genuinely
    held-out data without needing to retrain."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=["TransactionID", "isFraud"])
    y = df["isFraud"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def error_breakdown(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Cell 35 — confusion-matrix-style error type counts."""
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}


def tune_threshold(y_test: np.ndarray, y_proba: np.ndarray) -> dict:
    """Cell 36 — scan thresholds 0.1-0.85, pick the one that maximises F1."""
    results = []
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_t = (y_proba >= thresh).astype(int)
        if y_pred_t.sum() == 0:
            continue
        results.append({
            "threshold": round(float(thresh), 2),
            "precision": float(precision_score(y_test, y_pred_t, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_t)),
            "f1": float(f1_score(y_test, y_pred_t, zero_division=0)),
            "fraud_flagged": int(y_pred_t.sum()),
        })
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1"].idxmax()]
    return {"scan": results, "best": best_row.to_dict()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained fraud detection model.")
    parser.add_argument("--data", required=True, help="Path to df_final.csv")
    parser.add_argument("--model", required=True, help="Path to model.pkl")
    parser.add_argument("--out", default="models/", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60); print("LOADING MODEL + TEST SPLIT"); print("=" * 60)
    model = joblib.load(args.model)
    X_test, y_test = load_test_split(args.data)
    print(f"Test set: {X_test.shape}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print("\n" + "=" * 60); print("ERROR ANALYSIS (threshold=0.5)"); print("=" * 60)
    errors = error_breakdown(y_test.values, y_pred_default)
    for k, v in errors.items():
        print(f"  {k}: {v:,}")

    print("\n" + "=" * 60); print("THRESHOLD TUNING"); print("=" * 60)
    threshold_results = tune_threshold(y_test.values, y_proba)
    best = threshold_results["best"]
    print(f"Best threshold by F1: {best['threshold']}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1:        {best['f1']:.4f}")

    print("\n" + "=" * 60); print("QUALITY GATE"); print("=" * 60)
    pr_auc = average_precision_score(y_test, y_proba)
    recall_at_default = recall_score(y_test, y_pred_default)

    passed = pr_auc >= MIN_PR_AUC and recall_at_default >= MIN_RECALL_AT_DEFAULT_THRESHOLD
    print(f"PR-AUC:                     {pr_auc:.4f}  (minimum required: {MIN_PR_AUC})")
    print(f"Recall @ threshold=0.5:     {recall_at_default:.4f}  (minimum required: {MIN_RECALL_AT_DEFAULT_THRESHOLD})")
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'} — model is {'' if passed else 'NOT '}approved for deployment")

    report = {
        "pr_auc": float(pr_auc),
        "recall_at_0.5": float(recall_at_default),
        "gate_passed": bool(passed),
        "error_breakdown_at_0.5": errors,
        "recommended_threshold": best["threshold"],
        "threshold_scan": threshold_results["scan"],
    }
    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ evaluation_report.json saved -> {out_dir}/evaluation_report.json")


if __name__ == "__main__":
    main()
