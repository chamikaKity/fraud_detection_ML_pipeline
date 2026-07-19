"""
etl.py — Raw IEEE-CIS data -> cleaned df_clean.csv

This is the generic ETL stage of the pipeline: it owns everything that
happens BEFORE feature engineering. It mirrors Q1_EDA.ipynb cells 3 and
14-17 (missing values, outlier capping, encoding), minus the plotting/
exploration cells — those were for your own understanding, not something
a production pipeline needs to redo every run.

Model-specific work (the 6 engineered features, RF-based feature
selection) intentionally lives in feature_engineering.py, not here —
this stage doesn't touch isFraud at all, so its output (df_clean.csv)
stays reusable by any future model, not just this one.

Key design change vs. the notebook: every "fitted" decision (which
columns to drop, per-column medians, label encoders, outlier caps) is
SAVED to etl_artifacts.pkl. That file is what lets a single new
transaction be cleaned the same way later, in the serving API — you
can't recompute a median from one row.

Usage:
    python src/etl.py --raw-dir data/raw/ --out data/processed/
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MISSING_DROP_THRESHOLD = 90        # % missing -> column dropped entirely
FLAG_MISSING_THRESHOLD = 30        # % missing -> candidate for a "was_missing" flag
FLAG_CORR_THRESHOLD = 0.02         # |corr with isFraud| -> flag kept


def _read_csv_efficient(path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Read a wide CSV in chunks, downcasting each chunk's numeric dtypes
    (int64->smallest int, float64->float32) before appending it.

    A plain pd.read_csv() on the full file asks the C parser to
    materialise ~590K rows x ~394 columns at full float64/int64
    precision all at once — that single-shot peak is what was killing
    the process, not anything after the read completed (the log died
    mid-parse, before the shape print). Reading 100K rows at a time and
    shrinking each chunk immediately bounds peak memory to roughly one
    chunk's full-precision size, and the final concatenated frame is
    already downcasted, so it stays small too. No information is lost:
    downcast="integer"/"float" only steps down to a smaller dtype that
    still holds every value in that column exactly.
    """
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        int_cols = chunk.select_dtypes(include="int64").columns
        for col in int_cols:
            chunk[col] = pd.to_numeric(chunk[col], downcast="integer")

        float_cols = chunk.select_dtypes(include="float64").columns
        for col in float_cols:
            chunk[col] = pd.to_numeric(chunk[col], downcast="float")

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Load and merge the two raw IEEE-CIS tables — Cell 3."""
    raw_dir = Path(raw_dir)
    print("Loading transaction data...")
    train_trans = _read_csv_efficient(raw_dir / "train_transaction.csv")
    print(f"  train_transaction: {train_trans.shape}  "
          f"({train_trans.memory_usage(deep=True).sum() / 1024**2:.0f} MB in memory)")

    print("Loading identity data...")
    train_id = _read_csv_efficient(raw_dir / "train_identity.csv")
    print(f"  train_identity:    {train_id.shape}  "
          f"({train_id.memory_usage(deep=True).sum() / 1024**2:.0f} MB in memory)")

    df = train_trans.merge(train_id, on="TransactionID", how="left")
    print(f"  Merged dataset:    {df.shape}  "
          f"({df.memory_usage(deep=True).sum() / 1024**2:.0f} MB in memory)")
    return df


def drop_high_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Tier 1 — drop columns with >90% missing values — Cell 14-a."""
    missing_pct = df.isnull().mean() * 100
    drop_cols = missing_pct[missing_pct > MISSING_DROP_THRESHOLD].index.tolist()
    df_clean = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} columns (>{MISSING_DROP_THRESHOLD}% missing)")
    return df_clean, drop_cols


def add_missingness_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Add 'was_missing' flags for 30-90%-missing columns, keep only
    flags that correlate with isFraud — Cells 14-b, 14-c."""
    missing_pct = df.isnull().mean() * 100
    flag_candidates = missing_pct[
        (missing_pct > FLAG_MISSING_THRESHOLD) & (missing_pct <= MISSING_DROP_THRESHOLD)
    ].index.tolist()
    flag_candidates = [c for c in flag_candidates if c in df.columns]

    # Build every flag column in one dict, then concat ONCE — inserting
    # columns one at a time in a loop (the old version) forces pandas to
    # repeatedly reallocate and copy the whole dataframe internally,
    # which both triggers the "highly fragmented" warning and inflates
    # peak memory on a dataframe this wide. int8 instead of the default
    # int64 also cuts these flag columns' memory footprint by 8x — they
    # only ever hold 0 or 1, so int8 loses nothing.
    flag_data = {
        f"{col}_was_missing": df[col].isnull().astype(np.int8)
        for col in flag_candidates
    }
    df = pd.concat([df, pd.DataFrame(flag_data, index=df.index)], axis=1)

    flag_cols = list(flag_data.keys())
    flag_corr = df[flag_cols + ["isFraud"]].corr()["isFraud"].drop("isFraud")

    useful_flags = flag_corr[flag_corr.abs() > FLAG_CORR_THRESHOLD].index.tolist()
    weak_flags = [c for c in flag_cols if c not in useful_flags]

    df = df.drop(columns=weak_flags)
    print(f"Added {len(useful_flags)} missingness flags (dropped {len(weak_flags)} weak ones)")
    return df, useful_flags, weak_flags


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Median-impute numeric columns, mode-impute categorical columns
    — Cell 15. Returns the fitted medians/modes so the same values can
    be reused on new data later (a new transaction shouldn't be imputed
    with ITS OWN median of one row)."""
    exclude = ["TransactionID", "isFraud"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in exclude]

    medians, modes = {}, {}
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            medians[col] = df[col].median()
            df[col] = df[col].fillna(medians[col])
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            modes[col] = df[col].mode()[0]
            df[col] = df[col].fillna(modes[col])

    remaining = df.isnull().sum().sum()
    print(f"Imputed {len(medians)} numeric + {len(modes)} categorical columns "
          f"(remaining missing cells: {remaining})")
    return df, medians, modes


def treat_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Winsorize continuous columns at the 1st/99th percentile — Cell 16.
    Binary/flag columns (nunique <= 2) are skipped, nothing meaningful
    to cap. Returns the fitted bounds for reuse on new data."""
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ["TransactionID", "isFraud"]]
    continuous_cols = [c for c in num_cols if df[c].nunique() > 2]

    bounds = {}
    for col in continuous_cols:
        p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
        if p1 == p99:
            continue  # near-constant column, nothing to cap
        bounds[col] = (p1, p99)
        df[col] = df[col].clip(lower=p1, upper=p99)

    print(f"Winsorized {len(bounds)} continuous columns at [1%, 99%]")
    return df, bounds


def preserve_email_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Keep unencoded copies of the two email domain columns before
    LabelEncoder touches them. email_match (computed in
    feature_engineering.py) needs the raw strings — comparing two
    independently-fit LabelEncoder codes for "gmail.com" would give a
    false mismatch, since each column's encoder assigns codes separately."""
    df["P_emaildomain_raw"] = df["P_emaildomain"]
    df["R_emaildomain_raw"] = df["R_emaildomain"]
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode remaining categorical columns — Cell 17.
    Returns the fitted encoders so new data can be encoded identically."""
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c not in ["TransactionID", "P_emaildomain_raw", "R_emaildomain_raw"]]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"Label-encoded {len(cat_cols)} categorical columns")
    return df, encoders


def main():
    parser = argparse.ArgumentParser(description="Run the fraud detection ETL pipeline.")
    parser.add_argument("--raw-dir", required=True, help="Directory with train_transaction.csv / train_identity.csv")
    parser.add_argument("--out", default="data/processed/", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60); print("1. LOAD RAW DATA"); print("=" * 60)
    df_raw = load_raw_data(args.raw_dir)

    print("\n" + "=" * 60); print("2. MISSING VALUE HANDLING"); print("=" * 60)
    df_clean, drop_cols = drop_high_missing_columns(df_raw)
    df_clean, useful_flags, weak_flags = add_missingness_flags(df_clean)
    df_clean, medians, modes = impute_missing(df_clean)

    print("\n" + "=" * 60); print("3. OUTLIER TREATMENT"); print("=" * 60)
    df_clean, outlier_bounds = treat_outliers(df_clean)

    print("\n" + "=" * 60); print("4. ENCODE CATEGORICALS"); print("=" * 60)
    df_clean = preserve_email_raw(df_clean)
    df_clean, label_encoders = encode_categoricals(df_clean)

    # ── Save outputs ────────────────────────────────────────────
    # df_clean.csv is the ETL/ML-pipeline handoff point: generic cleaning
    # ends here, model-specific feature engineering picks up from this
    # file in feature_engineering.py.
    clean_path = out_dir / "df_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"\n✅ df_clean.csv saved: {df_clean.shape} -> {clean_path}")

    artifacts = {
        "drop_cols": drop_cols,
        "useful_flags": useful_flags,
        "weak_flags": weak_flags,
        "medians": medians,
        "modes": modes,
        "outlier_bounds": outlier_bounds,
        "label_encoders": label_encoders,
    }
    artifacts_path = out_dir / "etl_artifacts.pkl"
    joblib.dump(artifacts, artifacts_path)
    print(f"✅ etl_artifacts.pkl saved -> {artifacts_path}")
    print("   (these fitted values will be reused later to clean a single")
    print("    incoming transaction identically, in the serving API)")


if __name__ == "__main__":
    main()
