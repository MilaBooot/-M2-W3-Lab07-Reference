"""
src/models/predict.py
---------------------
Step 6 — Live Fraud Scoring (Inference)

Scores new, unseen insurance claims using the trained pipeline.
Accepts input as a CSV file and outputs a risk assessment table.

Usage:
    python -m src.models.predict --input test_claim.csv
    python -m src.models.predict --input claims_batch.csv

Output columns:
    Anomaly_Score   Continuous score. More negative = more suspicious.
    Flagged         1 = model flagged this claim for investigation.
    Risk_Level      CRITICAL / HIGH / MEDIUM / LOW

How the pipeline works at inference time:
    1. Raw CSV is loaded (no manual encoding or scaling needed).
    2. Categorical columns are one-hot encoded to match training format.
    3. Columns are aligned to the exact training feature set.
       Missing columns (unseen categories) are filled with 0.
    4. The pipeline applies StandardScaler then IsolationForest automatically.
    5. Anomaly score and risk level are returned.
"""

import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from src.data.utils import load_config, ordinal_days, ordinal_past_claims, ordinal_num_supplements

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def preprocess_input(df_raw):
    """Apply the same transformations as cleaning.py to new input records."""
    df = df_raw.copy()

    # Drop identifier columns if present
    df.drop(columns=["PolicyNumber", "RepNumber", "FraudFound_P"],
            inplace=True, errors="ignore")

    # Convert ordinal day ranges
    for col in ["Days_Policy_Accident", "Days_Policy_Claim"]:
        if col in df.columns:
            df[col] = df[col].apply(ordinal_days)

    # Convert past claims and supplements
    if "PastNumberOfClaims" in df.columns:
        df["PastNumberOfClaims"] = df["PastNumberOfClaims"].apply(ordinal_past_claims)
    if "NumberOfSuppliments" in df.columns:
        df["NumberOfSuppliments"] = df["NumberOfSuppliments"].apply(ordinal_num_supplements)

    # Encode binary columns
    for col in ["PoliceReportFiled", "WitnessPresent"]:
        if col in df.columns:
            df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)
    if "AgentType" in df.columns:
        df["AgentType"] = (df["AgentType"].str.strip().str.lower() == "internal").astype(int)

    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df


def predict(input_path, model_path="models/pipeline.pkl",
            columns_path="models/feature_columns.csv"):

    # ── Load pipeline ──────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run 'python -m src.models.train' first."
        )
    pipe = joblib.load(model_path)
    logging.info(f"Loaded pipeline from {model_path}")

    # Load the training feature column names for alignment
    train_cols = pd.read_csv(columns_path, header=None)[0].tolist()

    # ── Load and preprocess input ──────────────────────────────────
    df_raw = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df_raw)} claim(s) from {input_path}")

    df = preprocess_input(df_raw)

    # Align columns exactly to training feature set
    # New/unseen categories get filled with 0 (not present in training)
    df = df.reindex(columns=train_cols, fill_value=0)
    logging.info(f"Feature matrix aligned to {len(train_cols)} training columns")

    # ── Score ──────────────────────────────────────────────────────
    X_scaled = pipe.named_steps["scaler"].transform(df.values)
    scores   = pipe.named_steps["iso"].decision_function(X_scaled)
    preds    = pipe.predict(df.values)
    flagged  = (preds == -1).astype(int)

    # Assign risk level based on score thresholds
    def risk_level(score):
        if score < -0.10:
            return "CRITICAL"
        elif score < -0.05:
            return "HIGH"
        elif score < 0:
            return "MEDIUM"
        else:
            return "LOW"

    risk_levels = [risk_level(s) for s in scores]

    # ── Build output ───────────────────────────────────────────────
    result = df_raw.copy()
    result["Anomaly_Score"] = np.round(scores, 5)
    result["Flagged"]       = flagged
    result["Risk_Level"]    = risk_levels

    result_sorted = result.sort_values("Anomaly_Score").reset_index(drop=True)

    print("\n" + "=" * 65)
    print("  FRAUD RISK ASSESSMENT")
    print("=" * 65)
    display_cols = ["Anomaly_Score", "Flagged", "Risk_Level"]
    # Add a few key columns if available
    for col in ["Days_Policy_Accident", "Days_Policy_Claim",
                "PastNumberOfClaims", "VehiclePrice", "WitnessPresent",
                "PoliceReportFiled"]:
        if col in result_sorted.columns:
            display_cols.insert(-2, col)

    print(result_sorted[display_cols].to_string(index=True))
    print("=" * 65)
    print(f"\nClaims flagged for investigation: {flagged.sum()} / {len(flagged)}")
    print("Risk Level Guide:")
    print("  CRITICAL  score < -0.10  →  Immediate investigation")
    print("  HIGH      score < -0.05  →  Prioritise this week")
    print("  MEDIUM    score <  0.00  →  Add to investigation queue")
    print("  LOW       score ≥  0.00  →  Likely legitimate\n")

    return result_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score insurance claims for fraud risk.")
    parser.add_argument("--input",   required=True, help="Path to input CSV file")
    parser.add_argument("--model",   default="models/pipeline.pkl", help="Path to pipeline.pkl")
    parser.add_argument("--columns", default="models/feature_columns.csv",
                        help="Path to feature_columns.csv")
    args = parser.parse_args()

    predict(args.input, args.model, args.columns)
