"""
src/data/cleaning.py
--------------------
Step 3 — Data Cleaning Pipeline

Reads feature lists directly from config.yaml so no column names
are ever hardcoded. Transforms raw fraud_oracle.csv into a clean
numeric DataFrame ready for feature engineering.

Steps:
  1. Drop identifier columns listed under features.drop in config.
  2. Convert ordinal day-range strings to numeric midpoints.
  3. Convert PastNumberOfClaims and NumberOfSuppliments to numerics.
  4. Encode binary Yes/No and Internal/External columns as 0/1.
  5. Drop rows with missing values.
  6. Save cleaned_data.csv to data/processed/.

Run from project root:
    python -m src.data.cleaning
"""

import logging
import pandas as pd
from src.data.utils import (
    load_config, ordinal_days, ordinal_past_claims, ordinal_num_supplements
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def clean(cfg):
    raw_path     = cfg["data"]["raw_path"]
    cleaned_path = cfg["data"]["cleaned_path"]
    feat_cfg     = cfg["features"]

    # ── 1. Load ────────────────────────────────────────────────────
    df = pd.read_csv(raw_path)
    logging.info(f"Loaded raw data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    logging.info(f"Fraud rate in raw data: {df[feat_cfg['label']].mean():.2%}")

    # ── 2. Drop identifier columns (from config) ───────────────────
    drop_cols = feat_cfg.get("drop", [])
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    logging.info(f"Dropped identifier columns: {drop_cols}")

    # ── 3. Convert ordinal day-range strings → numeric midpoints ───
    # These two columns store time as string ranges like '1 to 7'.
    # We map them to numeric midpoints so the model can reason about
    # how quickly a claim was filed after the policy started.
    for col in ["Days_Policy_Accident", "Days_Policy_Claim"]:
        if col in df.columns:
            df[col] = df[col].apply(ordinal_days)
            logging.info(f"Converted {col} → numeric midpoints")

    # ── 4. Convert PastNumberOfClaims → numeric midpoints ──────────
    if "PastNumberOfClaims" in df.columns:
        df["PastNumberOfClaims"] = df["PastNumberOfClaims"].apply(ordinal_past_claims)
        logging.info("Converted PastNumberOfClaims → numeric midpoints")

    # ── 5. Convert NumberOfSuppliments → numeric midpoints ─────────
    if "NumberOfSuppliments" in df.columns:
        df["NumberOfSuppliments"] = df["NumberOfSuppliments"].apply(ordinal_num_supplements)
        logging.info("Converted NumberOfSuppliments → numeric midpoints")

    # ── 6. Encode binary columns (from config) → 0/1 ──────────────
    binary_cols = feat_cfg.get("binary", [])
    for col in binary_cols:
        if col not in df.columns:
            continue
        if col == "AgentType":
            df[col] = (df[col].str.strip().str.lower() == "internal").astype(int)
        else:
            # PoliceReportFiled, WitnessPresent: Yes=1, No=0
            df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)
    logging.info(f"Encoded binary columns as 0/1: {binary_cols}")

    # ── 7. Drop missing values ─────────────────────────────────────
    before_drop = len(df)
    df.dropna(inplace=True)
    dropped = before_drop - len(df)
    if dropped:
        logging.warning(f"Dropped {dropped} rows with missing values")

    # ── 8. Save ────────────────────────────────────────────────────
    df.to_csv(cleaned_path, index=False)
    logging.info(f"Saved cleaned data → {cleaned_path}")
    logging.info(f"Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    logging.info(f"Fraud rate after cleaning: {df[feat_cfg['label']].mean():.2%}")
    return df


if __name__ == "__main__":
    clean(load_config())
