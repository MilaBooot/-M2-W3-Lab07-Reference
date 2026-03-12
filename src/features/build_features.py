"""
src/features/build_features.py
-------------------------------
Step 4 — Feature Engineering

Reads the categorical column list directly from config.yaml —
no column names are hardcoded here.

Steps:
  1. Separate FraudFound_P label (read from config features.label).
  2. One-hot encode all categorical columns listed in config features.categorical.
  3. Reattach FraudFound_P as the last column for post-hoc evaluation.
  4. Save features.csv to data/processed/.

CRITICAL — Preprocessing Leakage:
    StandardScaler is NOT applied here. It will be fitted only on the
    feature matrix inside train.py. Fitting the scaler on the full
    dataset before any split would leak test-set statistics into training.

Run from project root:
    python -m src.features.build_features
"""

import logging
import pandas as pd
from src.data.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def build(cfg):
    cleaned_path  = cfg["data"]["cleaned_path"]
    features_path = cfg["data"]["features_path"]
    feat_cfg      = cfg["features"]
    label_col     = feat_cfg["label"]
    cat_cols      = feat_cfg["categorical"]

    df = pd.read_csv(cleaned_path)
    logging.info(f"Loaded cleaned data: {df.shape}")

    # ── 1. Separate the label ──────────────────────────────────────
    # IsolationForest is unsupervised — it never sees FraudFound_P
    # during training. We save it here and reattach at the end so
    # evaluation scripts can measure precision/recall against ground truth.
    y = df.pop(label_col)
    logging.info(f"Separated label '{label_col}' from feature matrix.")

    # ── 2. Confirm categorical columns from config exist ──────────
    present_cats = [c for c in cat_cols if c in df.columns]
    missing_cats = [c for c in cat_cols if c not in df.columns]
    if missing_cats:
        logging.warning(f"Categorical columns in config but not in data: {missing_cats}")
    logging.info(f"One-hot encoding {len(present_cats)} categorical columns.")

    # ── 3. One-hot encode ─────────────────────────────────────────
    # drop_first=False: keep all categories so the model sees the full
    # feature space. For Isolation Forest, multicollinearity is not
    # a concern — unlike linear models.
    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=present_cats, drop_first=False)
    after_cols = df.shape[1]
    logging.info(
        f"One-hot encoding expanded columns: {before_cols} → {after_cols}"
    )

    # ── 4. Reattach label as last column ──────────────────────────
    df[label_col] = y.values

    # ── 5. Save ───────────────────────────────────────────────────
    df.to_csv(features_path, index=False)
    logging.info(f"Saved feature matrix → {features_path}")
    logging.info(f"Final shape: {df.shape}  (label included as last column)")
    logging.info(f"Total training features (excl. label): {df.shape[1] - 1}")
    return df


if __name__ == "__main__":
    build(load_config())
