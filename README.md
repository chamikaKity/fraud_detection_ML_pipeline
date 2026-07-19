# Fraud Detection ML Pipeline

MSc Data Science coursework: fraud detection on the IEEE-CIS dataset using
supervised machine learning, with a full ETL → feature engineering →
training → evaluation → explainability pipeline.

## Structure

- `src/` — pipeline stages (`etl.py`, `feature_engineering.py`, `train.py`,
  `evaluate.py`, `explain.py`)
- `notebooks/` — Databricks notebooks, including `Q1_pipeline.ipynb`
  (entry point — runs the full pipeline end-to-end)
- `requirements.txt` / `requirements-optional.txt` — dependencies (the
  optional file includes `shap` for explainability)

## Running it

Open `notebooks/Q1_pipeline.ipynb` in Databricks and run all cells top to
bottom. Edit the paths in the Config section to point at your own raw data
location before running.
