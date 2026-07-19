"""
explain.py — SHAP explainability analysis for the trained fraud model.

Optional, run manually — not part of the automated pipeline. This is
for human interpretation (report writing, model audits), not something
downstream automation depends on.

Unlike Q1-5_explainability.ipynb (which retrains XGBoost from scratch
just to run SHAP on it), this script loads the already-trained
models/model.pkl directly — one of the practical benefits of having a
pipeline with saved artifacts instead of a notebook that starts fresh
each time.

Requires: pip install shap

Usage:
    python src/explain.py --data data/processed/df_final.csv --model models/model.pkl --out models/
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_test_sample(data_path: str, sample_size: int):
    """Reconstruct the same test split train.py used, then sample it —
    SHAP on the full test set is slow, so the notebook samples too."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=["TransactionID", "isFraud"])
    y = df["isFraud"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sample_idx = X_test.sample(n=min(sample_size, len(X_test)), random_state=42).index
    X_sample = X_test.loc[sample_idx].reset_index(drop=True)
    y_sample = y_test.loc[sample_idx].reset_index(drop=True)
    return X_sample, y_sample


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability analysis.")
    parser.add_argument("--data", required=True, help="Path to df_final.csv")
    parser.add_argument("--model", required=True, help="Path to model.pkl")
    parser.add_argument("--out", default="models/", help="Output directory")
    parser.add_argument("--sample-size", type=int, default=2000,
                         help="Number of test rows to run SHAP on (full test set is slow)")
    args = parser.parse_args()

    import shap  # imported here so the rest of the pipeline doesn't require it installed

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model + test sample...")
    model = joblib.load(args.model)
    X_sample, y_sample = load_test_sample(args.data, args.sample_size)
    print(f"SHAP sample: {X_sample.shape}")

    print("Computing SHAP values (this takes 1-2 minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_sample.columns).sort_values(ascending=False)

    print("\nTop 15 features by mean |SHAP value|:")
    print(importance.head(15).to_string())

    # Explain one actual fraud case, same as Cell 4 of the notebook
    fraud_positions = X_sample[y_sample == 1].index.tolist()
    example_case = None
    if fraud_positions:
        idx = fraud_positions[0]
        predicted_prob = model.predict_proba(X_sample.iloc[[idx]])[0, 1]
        example_case = {
            "row_index": int(idx),
            "predicted_probability": float(predicted_prob),
            "top_shap_contributions": pd.Series(
                shap_values[idx], index=X_sample.columns
            ).abs().sort_values(ascending=False).head(10).to_dict(),
        }
        print(f"\nExample fraud case (row {idx}): predicted probability {predicted_prob:.4f}")

    report = {
        "n_samples": len(X_sample),
        "base_value": float(explainer.expected_value),
        "top_features": importance.head(20).to_dict(),
        "example_fraud_case": example_case,
    }
    with open(out_dir / "shap_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ shap_report.json saved -> {out_dir}/shap_report.json")
    print("   (beeswarm/waterfall plots are for direct human review — regenerate")
    print("    those in a notebook when needed, rather than saving them from this script)")


if __name__ == "__main__":
    main()
