## Risk Note 1 — Class Imbalance
**Observation:** Only 6% of claims are fraudulent (923 out of 15,420).
**ML Impact:** Isolation Forest relies on anomalies being "rare and different."
  The 6% rate means the model must isolate a small minority — ideal for IF.
**Fix:** Set contamination=0.06 to match the known rate. Do not set it higher
  without evidence — this increases false positives and wastes investigator time.



## Risk Note — Contamination Leakage
**Observation:** Setting contamination by scanning all possible values against
  FraudFound_P on the full dataset is leakage — you are using the answer to
  tune the question.
**ML Impact:** The model will appear to perform well in development but will
  fail in production where labels do not exist.
**Fix:** Set contamination using external domain knowledge (industry fraud rate)
  or a held-out validation set with labels. Never tune it on the full dataset.
