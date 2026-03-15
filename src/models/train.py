"""
src/models/train.py
-------------------
Step 5 — Production Model Training

Reads all model parameters from config.yaml — nothing hardcoded.

Pipeline:
    StandardScaler  →  IsolationForest

Steps:
  1. Load features.csv. Separate label using config features.label.
  2. Fit Pipeline(StandardScaler + IsolationForest) with params from config.
  3. Score all claims. Evaluate against FraudFound_P ground truth.
  4. Export top_n_flagged most suspicious claims to models/top200_flagged.csv.
  5. Save serialised pipeline to models/pipeline.pkl.
  6. Save feature column names to models/feature_columns.csv.
  7. Write Precision / Recall / F1 to reports/metrics.txt.

Run from project root:
    python -m src.models.train
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)
from src.data.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def train(cfg):
    features_path = cfg["data"]["features_path"]
    metrics_path  = cfg["reports"]["metrics_path"]
    label_col     = cfg["features"]["label"]
    top_n         = cfg["model"]["top_n_flagged"]

    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # ── 1. Load features ──────────────────────────────────────────
    df = pd.read_csv(features_path)
    logging.info(f"Loaded feature matrix: {df.shape}")

    y    = df.pop(label_col).values
    X    = df.values
    cols = df.columns.tolist()
    logging.info(f"Features: {X.shape[1]}  |  Fraud rate: {y.mean():.2%}")

    # ── 2. Build and fit the pipeline ─────────────────────────────
    # All parameters come from config.yaml — change them there,
    # not here. This keeps the training script reusable across labs.
    model_cfg = cfg["model"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso",    IsolationForest(
            contamination = model_cfg["contamination"],
            n_estimators  = model_cfg["n_estimators"],
            max_samples   = model_cfg["max_samples"],
            random_state  = model_cfg["random_state"],
        ))
    ])

    logging.info("Fitting Pipeline(StandardScaler + IsolationForest)...")
    logging.info(f"  contamination = {model_cfg['contamination']}")
    logging.info(f"  n_estimators  = {model_cfg['n_estimators']}")
    logging.info(f"  max_samples   = {model_cfg['max_samples']}")

    pipe.fit(X)
    logging.info("Training complete.")

    # ── 3. Score all claims ────────────────────────────────────────
    # decision_function: negative = anomaly, more negative = more suspicious
    # predict: -1 = anomaly (flagged), 1 = normal (clean)
    X_scaled = pipe.named_steps["scaler"].transform(X)
    scores   = pipe.named_steps["iso"].decision_function(X_scaled)
    preds    = pipe.predict(X)
    flagged  = (preds == -1).astype(int)

    logging.info(f"Claims scored: {len(scores):,}")
    logging.info(f"Claims flagged: {flagged.sum():,} ({flagged.mean():.2%})")

    # ── 4. Evaluate against ground-truth labels ───────────────────
    precision = precision_score(y, flagged, zero_division=0)
    recall    = recall_score(y, flagged, zero_division=0)
    f1        = f1_score(y, flagged, zero_division=0)
    cm        = confusion_matrix(y, flagged)

    logging.info("─" * 50)
    logging.info(
        f"  Precision : {precision:.4f}  "
        f"({precision*100:.1f}% of flagged claims are actual fraud)"
    )
    logging.info(
        f"  Recall    : {recall:.4f}  "
        f"({recall*100:.1f}% of real fraud cases were caught)"
    )
    logging.info(f"  F1 Score  : {f1:.4f}")
    logging.info(f"  Confusion Matrix:\n{cm}")
    logging.info("─" * 50)
    logging.info(
        "Tip: Precision 0.15–0.25 and Recall 0.20–0.40 are realistic "
        "for unsupervised fraud detection without labelled training data."
    )

    # ── 5. Export ranked anomaly report ───────────────────────────
    raw_df = pd.read_csv(cfg["data"]["raw_path"])
    raw_df["Anomaly_Score"] = scores
    raw_df["Flagged"]       = flagged

    def assign_risk(score):
        if score < -0.10:   return "CRITICAL"
        elif score < -0.05: return "HIGH"
        elif score < 0:     return "MEDIUM"
        else:               return "LOW"

    raw_df["Risk_Level"] = [assign_risk(s) for s in scores]

    # Top N most suspicious for investigator queue
    top_flagged = (raw_df.sort_values("Anomaly_Score")
                         .head(top_n)
                         .reset_index(drop=True))
    top_flagged.to_csv("models/top200_flagged.csv", index=False)
    logging.info(
        f"Top {top_n} suspicious claims → models/top200_flagged.csv"
    )

    # Full scored dataset (for visualize.py plot 06)
    raw_df.sort_values("Anomaly_Score").to_csv(
        "models/all_scored.csv", index=False
    )

    # ── 6. Save pipeline and feature columns ──────────────────────
    joblib.dump(pipe, "models/pipeline.pkl")
    pd.Series(cols).to_csv(
        "models/feature_columns.csv", index=False, header=False
    )
    logging.info("Pipeline saved → models/pipeline.pkl")

    # ── 7. Write metrics ──────────────────────────────────────────
    metrics_text = (
        f"Isolation Forest — Insurance Fraud Detection\n"
        f"{'='*45}\n"
        f"Model type     : {model_cfg['type']}\n"
        f"contamination  : {model_cfg['contamination']}\n"
        f"n_estimators   : {model_cfg['n_estimators']}\n"
        f"max_samples    : {model_cfg['max_samples']}\n"
        f"random_state   : {model_cfg['random_state']}\n"
        f"{'─'*45}\n"
        f"Total claims   : {len(scores):,}\n"
        f"Flagged        : {flagged.sum():,}  ({flagged.mean():.2%})\n"
        f"Actual fraud   : {int(y.sum()):,}  ({y.mean():.2%})\n"
        f"{'─'*45}\n"
        f"Precision      : {precision:.4f}\n"
        f"Recall         : {recall:.4f}\n"
        f"F1 Score       : {f1:.4f}\n"
        f"{'─'*45}\n"
        f"Confusion Matrix:\n"
        f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}\n"
        f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}\n"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    logging.info(f"Metrics → {metrics_path}")
    print(metrics_text)
    return pipe, scores, flagged


if __name__ == "__main__":
    train(load_config())
