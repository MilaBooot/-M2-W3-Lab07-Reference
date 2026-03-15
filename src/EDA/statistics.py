"""
src/EDA/statistics.py
---------------------
Step 4 — Statistical Fraud Audit

Generates reports/statistics/statistical_summary.txt containing:
  1. Class imbalance stats (exact fraud count and rate).
  2. Mutual Information scores — which features know the most about fraud.
  3. Point-biserial correlations for numeric features vs FraudFound_P.
  4. Chi-squared p-values for categorical features vs FraudFound_P.

Run from project root:
    python -m src.EDA.statistics
"""

import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from src.data.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def run_statistics(cfg):
    cleaned_path  = cfg["data"]["cleaned_path"]
    stats_dir     = cfg["reports"]["statistics_dir"]
    os.makedirs(stats_dir, exist_ok=True)
    output_path   = os.path.join(stats_dir, "statistical_summary.txt")

    df = pd.read_csv(cleaned_path)
    y  = df["FraudFound_P"]
    logging.info(f"Loaded cleaned data: {df.shape}")

    lines = []
    lines.append("=" * 65)
    lines.append("  INSURANCE FRAUD DETECTION — STATISTICAL AUDIT REPORT")
    lines.append("=" * 65)
    lines.append("")

    # ── 1. Class Imbalance ─────────────────────────────────────────
    lines.append("── 1. CLASS IMBALANCE ──────────────────────────────────────")
    total      = len(df)
    n_fraud    = y.sum()
    n_legit    = total - n_fraud
    fraud_rate = y.mean()
    lines.append(f"  Total claims:          {total:>8,}")
    lines.append(f"  Legitimate (0):        {n_legit:>8,}  ({1-fraud_rate:.2%})")
    lines.append(f"  Fraudulent (1):        {n_fraud:>8,}  ({fraud_rate:.2%})")
    lines.append(f"  Imbalance ratio:       {n_legit/n_fraud:.1f}:1  (legitimate:fraud)")
    lines.append("")
    lines.append("  ML Impact: Isolation Forest handles imbalance naturally —")
    lines.append("  it does not use labels, so class skew does not bias training.")
    lines.append("  However, contamination should be set close to the fraud rate")
    lines.append(f"  ({fraud_rate:.3f}) to match the expected anomaly proportion.")
    lines.append("")

    # ── 2. Mutual Information ──────────────────────────────────────
    lines.append("── 2. MUTUAL INFORMATION SCORES (vs FraudFound_P) ─────────")
    lines.append("  Higher score = feature shares more information with the")
    lines.append("  fraud label. Use this to prioritise features.")
    lines.append("")

    # Encode any remaining categoricals for MI calculation
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include=["object", "string"]).columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    feature_cols = [c for c in df_enc.columns if c != "FraudFound_P"]
    X_enc = df_enc[feature_cols]

    mi_scores = mutual_info_classif(X_enc, y, random_state=42, discrete_features="auto")
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    lines.append(f"  {'Feature':<30}  {'MI Score':>10}")
    lines.append(f"  {'-'*30}  {'-'*10}")
    for feat, score in mi_series.items():
        marker = "  ← TOP SIGNAL" if score == mi_series.max() else ""
        lines.append(f"  {feat:<30}  {score:>10.4f}{marker}")
    lines.append("")

    # ── 3. Numeric Feature Correlations ───────────────────────────
    lines.append("── 3. POINT-BISERIAL CORRELATION (numeric features vs label) ")
    lines.append("  Measures linear relationship between each numeric feature")
    lines.append("  and the binary fraud label. Positive = more likely fraud.")
    lines.append("")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "FraudFound_P"]

    corr_results = []
    for col in numeric_cols:
        corr, pval = stats.pointbiserialr(y, df[col])
        corr_results.append((col, corr, pval))
    corr_results.sort(key=lambda x: abs(x[1]), reverse=True)

    lines.append(f"  {'Feature':<30}  {'Correlation':>12}  {'p-value':>10}")
    lines.append(f"  {'-'*30}  {'-'*12}  {'-'*10}")
    for col, corr, pval in corr_results:
        sig = " *" if pval < 0.05 else ""
        lines.append(f"  {col:<30}  {corr:>12.4f}  {pval:>10.4f}{sig}")
    lines.append("")
    lines.append("  * = statistically significant at p < 0.05")
    lines.append("")

    # ── 4. Categorical Chi-Squared Tests ──────────────────────────
    lines.append("── 4. CHI-SQUARED TEST (categorical features vs FraudFound_P)")
    lines.append("  Low p-value = the feature distribution differs significantly")
    lines.append("  between fraudulent and legitimate claims.")
    lines.append("")

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    chi2_results = []
    for col in cat_cols:
        ct = pd.crosstab(df[col], y)
        chi2, pval, dof, _ = stats.chi2_contingency(ct)
        chi2_results.append((col, chi2, pval, dof))
    chi2_results.sort(key=lambda x: x[1], reverse=True)

    lines.append(f"  {'Feature':<30}  {'Chi2':>10}  {'p-value':>10}  {'DoF':>5}")
    lines.append(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*5}")
    for col, chi2, pval, dof in chi2_results:
        sig = " *" if pval < 0.05 else ""
        lines.append(f"  {col:<30}  {chi2:>10.2f}  {pval:>10.4f}  {dof:>5}{sig}")
    lines.append("")
    lines.append("  * = statistically significant at p < 0.05")
    lines.append("")

    # ── 5. Top Fraud Signals Summary ──────────────────────────────
    lines.append("── 5. TOP FRAUD SIGNAL SUMMARY ─────────────────────────────")
    lines.append("  Features to prioritise in Isolation Forest feature matrix:")
    top5_mi = mi_series.head(5).index.tolist()
    for i, feat in enumerate(top5_mi, 1):
        lines.append(f"  {i}. {feat}  (MI = {mi_series[feat]:.4f})")
    lines.append("")
    lines.append("  Recommended contamination: "
                 f"{fraud_rate:.4f}  (matches actual fraud rate)")
    lines.append("")
    lines.append("=" * 65)

    # Write to file
    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    logging.info(f"Statistical summary saved → {output_path}")
    print(report)
    return mi_series


if __name__ == "__main__":
    cfg = load_config()
    run_statistics(cfg)
