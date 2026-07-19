"""
feature_engineering.py — df_clean.csv -> df_final.csv

This is the model-specific stage of the pipeline (Cells 19-22 of
Q1_EDA.ipynb), separated from the generic cleaning in etl.py. It starts
by re-loading df_clean.csv (Cell 18's output) — same as re-running the
notebook cell that loads it back in for the "Feature Engineering"
section.

Why this is its own script, not part of etl.py: it uses isFraud (the
label) to decide which features matter, and creates features that only
make sense for THIS fraud model. A different model trained on the same
df_clean.csv might engineer completely different features — so this
stage is kept separate and swappable, while etl.py stays reusable.

Usage:
    python src/feature_engineering.py --data data/processed/df_clean.csv --out data/processed/
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

N_TOP_FEATURES = 100  # final feature count (+ engineered features, always kept)
ENGINEERED_FEATURES = [
    "hour", "day_of_week", "amt_log",
    "card_avg_amt", "amt_vs_card_avg", "email_match",
]


def _read_csv_efficient(data_path: str, chunksize: int = 100_000) -> pd.DataFrame:
    """Read df_clean.csv in chunks, downcasting each chunk's numeric
    dtypes immediately — same fix as etl.py's loader, and needed for the
    same reason: df_clean.csv is plain text, so it doesn't remember that
    etl.py already downcasted these columns to float32/int8. A plain
    pd.read_csv() re-infers dtypes from scratch and reflates everything
    back to float64/int64, on a file that's now 641 columns wide (more
    than the original 394, after the missingness flags were added) —
    which is exactly what was killing this stage.
    """
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        int_cols = chunk.select_dtypes(include="int64").columns
        for col in int_cols:
            chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

        float_cols = chunk.select_dtypes(include="float64").columns
        for col in float_cols:
            chunk[col] = pd.to_numeric(chunk[col], downcast="float")

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_clean_data(data_path: str) -> pd.DataFrame:
    print(f"Loading {data_path} ...")
    df = _read_csv_efficient(data_path)
    print(f"  Loaded df_clean: {df.shape}  "
          f"({df.memory_usage(deep=True).sum() / 1024**2:.0f} MB in memory)")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create the 6 engineered features — Cell 19 + the corrected
    Feature 6 cell."""
    df["hour"] = ((df["TransactionDT"] / 3600) % 24).astype(int)
    df["day_of_week"] = ((df["TransactionDT"] / (3600 * 24)) % 7).astype(int)
    df["amt_log"] = np.log1p(df["TransactionAmt"])

    # card_avg_amt: mean TransactionAmt per card3 — a lookup table that
    # must be saved, since a single new transaction can't recompute a
    # groupby mean from itself at inference time.
    card_avg_lookup = df.groupby("card3")["TransactionAmt"].mean()
    df["card_avg_amt"] = df["card3"].map(card_avg_lookup)
    df["amt_vs_card_avg"] = df["TransactionAmt"] / (df["card_avg_amt"] + 1e-9)

    # email_match uses the raw string columns etl.py preserved specifically
    # for this step (P_emaildomain/R_emaildomain were label-encoded
    # independently, so comparing their encoded codes would be wrong).
    email_match = (
        (df["P_emaildomain_raw"] == df["R_emaildomain_raw"])
        & df["P_emaildomain_raw"].notna()
        & df["R_emaildomain_raw"].notna()
    )
    df["email_match"] = email_match.astype(int)
    df = df.drop(columns=["P_emaildomain_raw", "R_emaildomain_raw"])

    print(f"Engineered {len(ENGINEERED_FEATURES)} new features: {ENGINEERED_FEATURES}")
    return df, {"card_avg_lookup": card_avg_lookup}


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Rank features by Random Forest importance, keep the top N plus
    all engineered features — Cells 21-22."""
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    X_sample = sample_df.drop(columns=["TransactionID", "isFraud"])
    y_sample = sample_df["isFraud"]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_sample, y_sample)

    importance = pd.Series(rf.feature_importances_, index=X_sample.columns)
    top_features = importance.sort_values(ascending=False).head(N_TOP_FEATURES).index.tolist()

    for f in ENGINEERED_FEATURES:
        if f not in top_features:
            top_features.append(f)

    final_cols = ["TransactionID"] + top_features + ["isFraud"]
    df_final = df[final_cols].copy()

    print(f"Selected {len(top_features)} final features (top {N_TOP_FEATURES} by RF importance "
          f"+ engineered features)")
    return df_final, top_features


def main():
    parser = argparse.ArgumentParser(description="Feature engineering: df_clean.csv -> df_final.csv")
    parser.add_argument("--data", required=True, help="Path to df_clean.csv")
    parser.add_argument("--out", default="data/processed/", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60); print("1. LOAD CLEANED DATA"); print("=" * 60)
    df_clean = load_clean_data(args.data)

    print("\n" + "=" * 60); print("2. FEATURE ENGINEERING"); print("=" * 60)
    df_clean, feature_artifacts = engineer_features(df_clean)

    print("\n" + "=" * 60); print("3. FEATURE SELECTION"); print("=" * 60)
    df_final, top_features = select_features(df_clean)

    # ── Save outputs ────────────────────────────────────────────
    final_path = out_dir / "df_final.csv"
    df_final.to_csv(final_path, index=False)
    print(f"\n✅ df_final.csv saved: {df_final.shape} -> {final_path}")

    artifacts = {
        "card_avg_lookup": feature_artifacts["card_avg_lookup"],
        "top_features": top_features,
    }
    artifacts_path = out_dir / "feature_artifacts.pkl"
    joblib.dump(artifacts, artifacts_path)
    print(f"✅ feature_artifacts.pkl saved -> {artifacts_path}")
    print("   (needed later to engineer these same features for a single")
    print("    incoming transaction in the serving API)")


if __name__ == "__main__":
    main()
