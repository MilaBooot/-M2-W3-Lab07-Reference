"""
src/EDA/visualize.py
--------------------
Step 3 & 5 — Diagnostic Visualisations

Generates 7 plots saved to reports/figures/:

  Pre-model (run after cleaning.py):
    01_fraud_by_days_policy.png     — Fraud rate by Days_Policy_Accident
    02_fraud_by_past_claims.png     — Fraud rate by PastNumberOfClaims
    03_fraud_by_vehicle_price.png   — Fraud rate by VehiclePrice category
    04_age_distribution.png         — Age histogram: fraud vs legitimate
    05_witness_police_heatmap.png   — Fraud rate heatmap: WitnessPresent x PoliceReportFiled

  Post-model (run after train.py — skipped safely if scores not yet available):
    06_score_distribution.png       — Anomaly score histogram coloured by fraud label
    07_precision_recall_curve.png   — Precision & Recall vs contamination sweep

Run from project root:
    python -m src.EDA.visualize
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from src.data.utils import load_config

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Style ──────────────────────────────────────────────────────────
FRAUD_COLOR  = "#E74C3C"
LEGIT_COLOR  = "#2E86C1"
ACCENT_COLOR = "#1ABC9C"
BG_COLOR     = "#F8F9FA"
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})


def save(fig, path, name):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"Saved → {path}")


# ── Plot 01: Fraud rate by Days_Policy_Accident ────────────────────
def plot_01(df, figures_dir):
    path = os.path.join(figures_dir, "01_fraud_by_days_policy.png")

    # Map numeric midpoints back to readable labels for the x-axis
    day_labels = {0: "none\n(0 days)", 4: "1–7 days", 11: "8–15 days",
                  22: "15–30 days", 35: "30+ days"}
    df2 = df.copy()
    df2["Days_Label"] = df2["Days_Policy_Accident"].map(day_labels).fillna("other")
    order = ["none\n(0 days)", "1–7 days", "8–15 days", "15–30 days", "30+ days"]

    fraud_rate = (df2.groupby("Days_Label")["FraudFound_P"]
                    .mean()
                    .reindex(order)
                    .fillna(0) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, fraud_rate.values,
                  color=[FRAUD_COLOR if v == fraud_rate.max() else LEGIT_COLOR
                         for v in fraud_rate.values],
                  edgecolor="white", linewidth=0.8)

    # Annotate each bar
    for bar, val in zip(bars, fraud_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(df["FraudFound_P"].mean() * 100, color="grey",
               linestyle="--", linewidth=1, label=f"Overall fraud rate ({df['FraudFound_P'].mean()*100:.1f}%)")
    ax.set_title("Fraud Rate by Days Between Policy Start and Accident\n"
                 "→ Claims filed within 1–7 days are the highest-risk group",
                 fontweight="bold")
    ax.set_xlabel("Days Between Policy Start and Accident")
    ax.set_ylabel("Fraud Rate (%)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, fraud_rate.max() * 1.3)
    save(fig, path, "01")


# ── Plot 02: Fraud rate by PastNumberOfClaims ──────────────────────
def plot_02(df, figures_dir):
    path = os.path.join(figures_dir, "02_fraud_by_past_claims.png")

    label_map = {0: "none", 1: "1", 3: "2–4", 5: "4+"}
    df2 = df.copy()
    df2["Claims_Label"] = df2["PastNumberOfClaims"].map(label_map).fillna("other")
    order = ["none", "1", "2–4", "4+"]

    fraud_rate = (df2.groupby("Claims_Label")["FraudFound_P"]
                    .mean()
                    .reindex(order)
                    .fillna(0) * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(order, fraud_rate.values,
                  color=[FRAUD_COLOR if v == fraud_rate.max() else LEGIT_COLOR
                         for v in fraud_rate.values],
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, fraud_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(df["FraudFound_P"].mean() * 100, color="grey",
               linestyle="--", linewidth=1, label="Overall fraud rate")
    ax.set_title("Fraud Rate by Number of Past Claims\n"
                 "→ Serial claimants (4+) show significantly higher fraud rates",
                 fontweight="bold")
    ax.set_xlabel("Past Number of Claims")
    ax.set_ylabel("Fraud Rate (%)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, fraud_rate.max() * 1.3)
    save(fig, path, "02")


# ── Plot 03: Fraud rate by VehiclePrice ───────────────────────────
def plot_03(df, figures_dir):
    path = os.path.join(figures_dir, "03_fraud_by_vehicle_price.png")

    price_order = ["less than 20000", "20000 to 29000", "30000 to 39000",
                   "40000 to 59000",  "60000 to 69000", "more than 69000"]
    price_labels = ["<20k", "20–29k", "30–39k", "40–59k", "60–69k", ">69k"]

    fraud_rate = (df.groupby("VehiclePrice")["FraudFound_P"]
                    .mean()
                    .reindex(price_order)
                    .fillna(0) * 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(price_labels, fraud_rate.values,
                  color=[FRAUD_COLOR if v == fraud_rate.max() else LEGIT_COLOR
                         for v in fraud_rate.values],
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, fraud_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(df["FraudFound_P"].mean() * 100, color="grey",
               linestyle="--", linewidth=1, label="Overall fraud rate")
    ax.set_title("Fraud Rate by Vehicle Price Category\n"
                 "→ High-value vehicles (>$69k) are disproportionately involved in fraud",
                 fontweight="bold")
    ax.set_xlabel("Vehicle Price Range")
    ax.set_ylabel("Fraud Rate (%)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, fraud_rate.max() * 1.3)
    save(fig, path, "03")


# ── Plot 04: Age distribution fraud vs legitimate ──────────────────
def plot_04(df, figures_dir):
    path = os.path.join(figures_dir, "04_age_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    fraud = df[df.FraudFound_P == 1]["Age"]
    legit = df[df.FraudFound_P == 0]["Age"]

    ax.hist(legit, bins=30, alpha=0.6, color=LEGIT_COLOR,
            label=f"Legitimate ({len(legit):,})", edgecolor="white")
    ax.hist(fraud, bins=30, alpha=0.8, color=FRAUD_COLOR,
            label=f"Fraudulent ({len(fraud):,})", edgecolor="white")

    ax.axvline(fraud.mean(), color=FRAUD_COLOR, linestyle="--", linewidth=1.5,
               label=f"Fraud mean age: {fraud.mean():.1f}")
    ax.axvline(legit.mean(), color=LEGIT_COLOR, linestyle="--", linewidth=1.5,
               label=f"Legit mean age:  {legit.mean():.1f}")

    ax.set_title("Age Distribution: Fraudulent vs Legitimate Claims\n"
                 "→ Checks whether fraud concentrates in specific age groups",
                 fontweight="bold")
    ax.set_xlabel("Policyholder Age")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    save(fig, path, "04")


# ── Plot 05: Witness × Police Report heatmap ──────────────────────
def plot_05(df, figures_dir):
    path = os.path.join(figures_dir, "05_witness_police_heatmap.png")

    df2 = df.copy()
    df2["WitnessPresent_Label"]    = df2["WitnessPresent"].map({1: "Witness: Yes", 0: "Witness: No"})
    df2["PoliceReportFiled_Label"] = df2["PoliceReportFiled"].map({1: "Police: Yes", 0: "Police: No"})

    pivot = df2.pivot_table(
        values="FraudFound_P",
        index="WitnessPresent_Label",
        columns="PoliceReportFiled_Label",
        aggfunc="mean"
    ) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Fraud Rate (%)"},
                vmin=0, vmax=pivot.max().max())
    ax.set_title("Fraud Rate (%) by Witness Presence and Police Report\n"
                 "→ No witness + no police report = highest fraud concentration",
                 fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, path, "05")


# ── Plot 06: Anomaly score distribution ───────────────────────────
def plot_06(df_raw, scores, figures_dir):
    path = os.path.join(figures_dir, "06_score_distribution.png")

    fraud_scores = scores[df_raw["FraudFound_P"] == 1]
    legit_scores = scores[df_raw["FraudFound_P"] == 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(legit_scores, bins=60, alpha=0.6, color=LEGIT_COLOR,
            label=f"Legitimate ({len(legit_scores):,})", density=True)
    ax.hist(fraud_scores, bins=60, alpha=0.8, color=FRAUD_COLOR,
            label=f"Fraudulent ({len(fraud_scores):,})", density=True)

    threshold = np.percentile(scores, 6)  # approx contamination=0.06 threshold
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Decision threshold ≈ {threshold:.3f}")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1, label="Score = 0")

    ax.set_title("Anomaly Score Distribution: Fraudulent vs Legitimate Claims\n"
                 "→ Fraudulent claims cluster in the left tail (most negative scores)",
                 fontweight="bold")
    ax.set_xlabel("Anomaly Score  (more negative = more suspicious)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    save(fig, path, "06")


# ── Plot 07: Precision & Recall vs contamination ──────────────────
def plot_07(contam_results, figures_dir):
    path = os.path.join(figures_dir, "07_precision_recall_curve.png")

    df_r = pd.DataFrame(contam_results)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_r["contamination"], df_r["precision"] * 100,
            "o-", color=LEGIT_COLOR, linewidth=2, label="Precision")
    ax.plot(df_r["contamination"], df_r["recall"] * 100,
            "s-", color=FRAUD_COLOR, linewidth=2, label="Recall")
    ax.plot(df_r["contamination"], df_r["f1"] * 100,
            "^--", color=ACCENT_COLOR, linewidth=2, label="F1 Score")

    ax.axvline(0.06, color="grey", linestyle=":", linewidth=1.5,
               label="contamination = 0.06 (recommended)")

    ax.set_title("Precision, Recall & F1 vs Contamination Setting\n"
                 "→ Higher contamination = more claims flagged = higher recall, lower precision",
                 fontweight="bold")
    ax.set_xlabel("Contamination (fraction of data flagged as anomalies)")
    ax.set_ylabel("Score (%)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, df_r["contamination"].max() + 0.01)
    save(fig, path, "07")


# ── Main orchestrator ──────────────────────────────────────────────
def run_visualizations(cfg):
    cleaned_path = cfg["data"]["cleaned_path"]
    features_path = cfg["data"]["features_path"]
    figures_dir  = cfg["reports"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Load cleaned data (required for plots 01–05)
    df = pd.read_csv(cleaned_path)
    logging.info(f"Loaded cleaned data: {df.shape}")

    plot_01(df, figures_dir)
    plot_02(df, figures_dir)
    plot_03(df, figures_dir)
    plot_04(df, figures_dir)
    plot_05(df, figures_dir)
    logging.info("Plots 01–05 saved.")

    # Plots 06 and 07 require the trained model — skip gracefully if not ready
    model_path = "models/pipeline.pkl"
    if not os.path.exists(model_path):
        logging.warning("models/pipeline.pkl not found — skipping plots 06 and 07.")
        logging.warning("Re-run this script after completing Step 5 (train.py).")
        return

    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_score, recall_score, f1_score

    pipe    = joblib.load(model_path)
    df_feat = pd.read_csv(features_path)
    y       = df_feat.pop("FraudFound_P").values
    X_scaled = pipe.named_steps["scaler"].transform(df_feat.values)
    scores   = pipe.named_steps["iso"].decision_function(X_scaled)

    plot_06(df, scores, figures_dir)

    # Contamination sweep for plot 07
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler().fit(df_feat.values)
    X_s     = scaler.transform(df_feat.values)
    contam_results = []
    for c in [0.01, 0.03, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        iso  = IsolationForest(contamination=c, n_estimators=100, random_state=42)
        iso.fit(X_s)
        pred = (iso.predict(X_s) == -1).astype(int)
        contam_results.append({
            "contamination": c,
            "flagged":       pred.sum(),
            "precision":     precision_score(y, pred, zero_division=0),
            "recall":        recall_score(y, pred, zero_division=0),
            "f1":            f1_score(y, pred, zero_division=0),
        })

    plot_07(contam_results, figures_dir)
    logging.info("All 7 plots saved to reports/figures/")


if __name__ == "__main__":
    cfg = load_config()
    run_visualizations(cfg)
