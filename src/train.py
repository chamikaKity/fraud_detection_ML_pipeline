"""
train.py — Training script for the fraud detection model.

This is stage 3 of the pipeline:
  1. etl.py                 — raw data -> df_clean.csv (generic cleaning)
  2. feature_engineering.py — df_clean.csv -> df_final.csv (model-specific)
  3. train.py                — df_final.csv -> trained model  (this script)

Trains the TUNED XGBoost model — this is the actual best-performing model
per Q1-3_models.ipynb / Q1_report.md, not the default-hyperparameter version.

Design note: by default this script does NOT re-run the hyperparameter
search (RandomizedSearchCV) that originally found these values — that
search took several minutes and is a one-off research step, not
something a routine "retrain on fresh data" pipeline run should repeat
every time. Instead it reuses the best hyperparameters already found
(BEST_XGB_PARAMS below, copied from the notebook's search output).
Pass --tune if you deliberately want to re-run the search (e.g. after
enough new data has arrived that the old hyperparameters might be stale).

Usage:
    python src/etl.py                 --raw-dir data/raw/                --out data/processed/
    python src/feature_engineering.py --data data/processed/df_clean.csv --out data/processed/
    python src/train.py               --data data/processed/df_final.csv --out models/
    python src/train.py               --data data/processed/df_final.csv --out models/ --tune
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from scipy.stats import uniform, randint
from xgboost import XGBClassifier

# Best hyperparameters found via RandomizedSearchCV in Q1-3_models.ipynb
# (Cell 32, PART 1: XGBoost Tuning — CV PR-AUC 0.5300)
BEST_XGB_PARAMS = {
    "n_estimators": 484,
    "max_depth": 8,
    "learning_rate": 0.08125956761539498,
    "subsample": 0.6911740650167767,
    "colsample_bytree": 0.7297380084021096,
    "min_child_weight": 1,
    "gamma": 0.061043977350336676,
}


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load df_final.csv and split into features/target — Cell 24."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=["TransactionID", "isFraud"])
    y = df["isFraud"]
    return X, y


def search_best_params(X_train, y_train, scale_pos_weight: float) -> dict:
    """Re-run the hyperparameter search — Cell 32, PART 1. Only used
    when --tune is passed; slow (several minutes)."""
    sample_idx = pd.Series(range(len(X_train))).sample(n=50000, random_state=42)
    X_tune = X_train.iloc[sample_idx].reset_index(drop=True)
    y_tune = y_train.iloc[sample_idx].reset_index(drop=True)

    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
    }
    base = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=20,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42, n_jobs=-1, verbose=1,
    )
    print("Running RandomizedSearchCV (this will take a few minutes)...")
    start = time.time()
    search.fit(X_tune, y_tune)
    print(f"Search complete in {time.time() - start:.1f}s — best CV PR-AUC: {search.best_score_:.4f}")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    return search.best_params_


def train(X_train, y_train, params: dict, scale_pos_weight: float) -> XGBClassifier:
    """Train the tuned XGBoost model — Cell 32's 'Retrain best XGBoost' step."""
    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, threshold: float = 0.75) -> dict:
    """Compute the same metrics used in the coursework report."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the tuned fraud detection XGBoost model.")
    parser.add_argument("--data", required=True, help="Path to df_final.csv")
    parser.add_argument("--out", default="models/", help="Output directory for artifacts")
    parser.add_argument("--threshold", type=float, default=0.75, help="Classification threshold")
    parser.add_argument("--tune", action="store_true",
                         help="Re-run RandomizedSearchCV instead of using saved best params (slow)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60); print("LOADING DATA"); print("=" * 60)
    X, y = load_data(args.data)
    print(f"Loaded {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Fraud rate: {y.mean()*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)  # fit on train only — avoids leakage into test
    # kept for parity with the notebook / other models (LR, KNN, ANN);
    # XGBoost itself doesn't need scaled inputs.

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    if args.tune:
        print("\n" + "=" * 60); print("HYPERPARAMETER SEARCH (--tune passed)"); print("=" * 60)
        params = search_best_params(X_train, y_train, scale_pos_weight)
    else:
        print("\nUsing saved best hyperparameters (pass --tune to re-search):")
        for k, v in BEST_XGB_PARAMS.items():
            print(f"  {k}: {v}")
        params = BEST_XGB_PARAMS

    print("\n" + "=" * 60); print("TRAINING"); print("=" * 60)
    start = time.time()
    model = train(X_train, y_train, params, scale_pos_weight)
    train_time = time.time() - start
    print(f"Training complete in {train_time:.1f}s")

    print("\n" + "=" * 60); print("EVALUATION"); print("=" * 60)
    metrics = evaluate(model, X_test, y_test, threshold=args.threshold)
    metrics["train_time_seconds"] = round(train_time, 1)
    metrics["n_train_rows"] = int(len(X_train))
    metrics["n_test_rows"] = int(len(X_test))
    metrics["hyperparameters"] = params
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Save artifacts ──────────────────────────────────────────
    joblib.dump(model, out_dir / "model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\n✅ Artifacts saved to {out_dir}/")
    print("   model.pkl, scaler.pkl, feature_names.json, metrics.json")


if __name__ == "__main__":
    main()
