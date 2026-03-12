# M2-Lab07 — Isolation Forest: Insurance Claim Fraud Detection

An unsupervised anomaly detection pipeline that scores every vehicle insurance claim by suspicion level, routing the most anomalous records to a fraud investigation team — without any labelled training data.

---

## Business Problem

Auto insurance fraud costs US insurers an estimated **$40 billion annually**. A fraud investigation team has capacity to manually review only ~200 claims per week, but thousands arrive in the backlog. The challenge: which 200 do you investigate first?

This project builds an **Isolation Forest** pipeline that assigns a continuous anomaly score to every claim. Investigators work through the ranked list from most suspicious to least — maximising fraud recovery per hour of investigator time.

**Why Isolation Forest?**
Unlike supervised models (XGBoost, Logistic Regression), Isolation Forest requires **no labelled training data**. It detects fraud purely by identifying claims that are statistically unusual — "few and different" from the normal population. This makes it deployable immediately, before any labelled fraud dataset has been assembled.

---

## Project Philosophy: Discovery First

Following industry best practices, this project is structured in two phases:

1. **Discovery (EDA Phase):** Interactive exploration to understand the fraud pattern, identify key signals, and determine the right contamination setting.
2. **Deployment (Pipeline Phase):** Modularised scripts for cleaning, feature engineering, training, and live inference.

---

## Project Structure

```
├── configs/
│   └── config.yaml              # Centralised paths, feature lists, model parameters
├── data/
│   ├── raw/                     # fraud_oracle.csv — immutable source (gitignored)
│   └── processed/               # cleaned_data.csv, features.csv (gitignored)
├── models/                      # pipeline.pkl, top200_flagged.csv (gitignored)
├── notebooks/
│   └── eda_fraud.ipynb          # EDA, fraud profiling, contamination experiment
├── reports/
│   ├── figures/                 # 7 diagnostic plots (.png)
│   ├── statistics/              # statistical_summary.txt
│   ├── metrics.txt              # Precision, Recall, F1
│   └── risk_notes.md            # Risk analysis in Observation → Impact → Fix format
├── src/
│   ├── data/
│   │   ├── utils.py             # load_config(), ordinal_days(), and other helpers
│   │   └── cleaning.py          # Drops IDs, converts ordinals, encodes binary cols
│   ├── EDA/
│   │   ├── statistics.py        # Mutual Information, chi-squared, correlation audit
│   │   └── visualize.py         # All 7 diagnostic plots
│   ├── features/
│   │   └── build_features.py    # One-hot encoding, label separation
│   └── models/
│       ├── train.py             # Pipeline fit, scoring, metrics, ranked export
│       └── predict.py           # Live inference on new claims
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and navigate
git clone <your-classroom-repo-url>
cd M2-Lab07-Isolation-Forest

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Download fraud_oracle.csv from Google Classroom
# Move it to: data/raw/fraud_oracle.csv

# 5. Run the full pipeline in order
python -m src.data.cleaning
python -m src.EDA.statistics
python -m src.EDA.visualize                         # generates plots 01–05
python -m src.features.build_features
python -m src.models.train
python -m src.EDA.visualize                         # generates plots 06–07

# 6. Score a new claim
python -m src.models.predict --input test_claim.csv
```

---

## Configuration

All paths, feature lists, and model parameters live in **`configs/config.yaml`**. No column names or file paths are hardcoded in any script.

```yaml
model:
  type:          "IsolationForest"
  contamination: 0.06      # change this to tune how many claims get flagged
  n_estimators:  100
  max_samples:   256
  top_n_flagged: 200
```

**Key parameters to understand:**

| Parameter | What it controls | When to change it |
|-----------|-----------------|-------------------|
| `contamination` | Fraction of claims flagged as anomalies | Increase if recall is too low; decrease if too many false positives |
| `n_estimators` | Number of isolation trees | Higher = more stable scores; rarely needs changing above 100 |
| `max_samples` | Samples per tree | 256 is empirically optimal per the original IF paper |
| `top_n_flagged` | Size of investigator queue | Match to team's weekly review capacity |

---

## Dataset

**Vehicle Insurance Claim Fraud Detection**
- Source: [Kaggle — Shivam Bansal](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- Records: 15,420 claims
- Fraud rate: ~6% (approximately 923 fraudulent claims)
- Features: 33 columns covering policy details, vehicle info, claim behaviour, and a fraud label

Download `fraud_oracle.csv` from Google Classroom and place it in `data/raw/`.

**Key fraud signals identified in EDA:**

| Feature | Fraud pattern |
|---------|--------------|
| `Days_Policy_Accident` | Claims filed within 1–7 days of policy start are highest risk |
| `PastNumberOfClaims` | Policyholders with 4+ past claims show disproportionate fraud |
| `VehiclePrice` | High-value vehicles (>$69k) are over-represented in fraud |
| `WitnessPresent` | Fraudulent claims are 3× more likely to have no witness |
| `PoliceReportFiled` | Fraudulent claims are 3× more likely to have no police report |
| `AddressChange_Claim` | Address changed near claim date correlates strongly with fraud |

---

## Key Features

- **Config-driven pipeline:** All column names and parameters in `config.yaml` — swap datasets by editing one file.
- **Leakage-free design:** `StandardScaler` is fitted inside the pipeline on the feature matrix only — never on the raw or full dataset.
- **Label separation:** `FraudFound_P` is popped from the feature matrix before any model sees it, then reattached for post-hoc evaluation only.
- **Ranked investigator output:** `models/top200_flagged.csv` lists the 200 most suspicious claims sorted by anomaly score, ready to route to investigators.
- **Risk levels:** Each claim is assigned `CRITICAL / HIGH / MEDIUM / LOW` based on anomaly score thresholds.
- **Contamination experiment:** `notebooks/eda_fraud.ipynb` demonstrates the effect of tuning contamination on precision/recall and proves that fitting contamination to known labels is a form of leakage.
- **7 diagnostic plots:** EDA plots 01–05 characterise the fraud pattern. Post-model plots 06–07 show score distribution and the precision-recall tradeoff across contamination values.

---

## Pipeline Execution Order

```
src/data/cleaning.py        →  data/processed/cleaned_data.csv
src/EDA/statistics.py       →  reports/statistics/statistical_summary.txt
src/EDA/visualize.py        →  reports/figures/01–05_*.png
src/features/build_features.py  →  data/processed/features.csv
src/models/train.py         →  models/pipeline.pkl
                               models/top200_flagged.csv
                               reports/metrics.txt
src/EDA/visualize.py        →  reports/figures/06–07_*.png  (post-model)
src/models/predict.py       →  live risk assessment on new claims
```

---

## Outputs

| File | Description |
|------|-------------|
| `models/pipeline.pkl` | Serialised `Pipeline(StandardScaler + IsolationForest)` |
| `models/top200_flagged.csv` | 200 most suspicious claims for investigator review |
| `models/feature_columns.csv` | Training feature names used to align inference inputs |
| `reports/metrics.txt` | Precision, Recall, F1 at `contamination=0.06` |
| `reports/statistics/statistical_summary.txt` | Mutual Information, correlations, chi-squared results |
| `reports/figures/` | 7 diagnostic `.png` plots |
| `reports/risk_notes.md` | Risk analysis document (student-authored) |

---

## Live Inference

Score a new claim from a CSV file:

```bash
python -m src.models.predict --input test_claim.csv
```

Output:

```
=================================================================
  FRAUD RISK ASSESSMENT
=================================================================
   Anomaly_Score  Days_Policy_Accident  ...  Flagged  Risk_Level
0        -0.082                1 to 7   ...        1    CRITICAL
=================================================================

Risk Level Guide:
  CRITICAL  score < -0.10  →  Immediate investigation
  HIGH      score < -0.05  →  Prioritise this week
  MEDIUM    score <  0.00  →  Add to investigation queue
  LOW       score >=  0.00  →  Likely legitimate
```

The `pipeline.pkl` handles all preprocessing automatically — no manual encoding or scaling needed at inference time.

---

## Design Notes

- **No train/test split:** Isolation Forest is unsupervised — it scores all records. There is no target variable to leak through a split.
- **StandardScaler inside the pipeline:** This ensures the scaler is always applied before IsolationForest at both training and inference time. Forgetting to scale at inference is a common production bug that this design prevents.
- **Contamination is not a free parameter:** Setting contamination by scanning values against `FraudFound_P` on the full dataset is leakage. Use domain knowledge (industry fraud rate) or a separate held-out validation set with labels.
- **Consistency across labs:** This project uses the same folder structure, `config.yaml` pattern, and `src/` layout as all other Module 2 labs. Keeping this structure identical makes production deployment straightforward.

---

## Suggested Next Steps

- Add unit tests for `src/data/utils.py` and `src/features/build_features.py`.
- Add CI (GitHub Actions) to lint and test on every push.
- Experiment with `n_estimators=200` and compare F1 stability across multiple random seeds.
- Try combining Isolation Forest scores with a supervised XGBoost model (from Lab05) for a hybrid ensemble approach.
- Tune `contamination` using a held-out labelled validation set rather than the full dataset.
