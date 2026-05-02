# 🩺 Pima Diabetes Prediction — Leakage-Safe ML Pipeline

A clean Kaggle + GitHub machine-learning project for the **Pima Diabetes** benchmark, focused on leakage-safe preprocessing, model comparison, calibrated probabilities, validation-based threshold selection, and a reusable scoring artifact.

> ⚕️ **Scope:** this is a benchmark ML workflow for reproducible analysis and model evaluation. It is **not** a clinical recommendation or medical decision system.

---

## ✨ What this repo provides

- ✅ **Publish-ready notebook** with EDA, validation-safe preprocessing, model comparison, calibration, and final test evaluation
- 🔒 **Leakage-safe pipeline**: zero handling, feature engineering, imputation, scaling, optional SMOTE/fallback balancing, and modeling stay inside the training pipeline
- 📊 **Decision-aware evaluation** using ROC-AUC, AUPRC, precision, recall, F1, Brier score, confusion matrices, calibration, and threshold analysis
- 🎯 **Validation-selected operating threshold** based on an illustrative false-positive / false-negative cost policy
- 📦 **Exportable artifact bundle** for inference with a matching CLI scoring script
- 🧪 **Kaggle-friendly fallback**: if `imbalanced-learn` is unavailable or incompatible, the notebook falls back to estimator-side imbalance handling instead of failing at import time

---

## 📁 Repository structure

```text
.
├── diabetes-prediction-ml-pipeline.ipynb   # Main notebook
├── scripts/
│   └── predict.py                          # Optional CSV scoring script
├── artifacts/
│   └── README.md                           # Generated artifacts live here
├── data/
│   └── raw/
│       └── README.md                       # Place diabetes.csv here for local runs
├── CASE_STUDY.md                           # Technical case-study summary
├── requirements.txt                        # Python dependencies
├── LICENSE
└── README.md
```

---

## 📚 Dataset

Dataset: **Pima Indians Diabetes Database** / **Pima Diabetes** benchmark.

Expected file name for local runs:

```text
data/raw/diabetes.csv
```

Expected columns:

| Column | Meaning |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skinfold thickness |
| `Insulin` | 2-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |
| `Outcome` | `1 = diabetes-positive`, `0 = diabetes-negative` |

The notebook also searches common Kaggle input paths and validates the schema before modeling.

---

## 🚀 Quick start

### 1) Install dependencies

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
```

### 2) Add the data

Place the CSV here:

```text
data/raw/diabetes.csv
```

### 3) Run the notebook

Open and run:

```text
diabetes-prediction-ml-pipeline.ipynb
```

The notebook will:

- inspect schema, class balance, invalid zero values, and feature distributions
- train baseline and tuned models using a leakage-safe pipeline
- calibrate probabilities
- select an operating threshold on validation data only
- evaluate once on the held-out test set
- export a reusable artifact bundle to `artifacts/pima_best_pipeline.joblib`

---

## 🧠 Modeling workflow

```text
EDA
→ leakage-safe preprocessing
→ model comparison
→ hyperparameter tuning
→ probability calibration
→ validation-selected threshold
→ held-out test evaluation
→ interpretation
→ exportable artifact
```

Key safeguards:

- preprocessing is fitted inside cross-validation and training folds
- threshold selection is performed on validation data only
- final test results are reported once after model and threshold selection
- real-world medical use is explicitly out of scope without external validation and governance

---

## 📦 Artifacts

When `SAVE_ARTIFACTS = True`, the notebook exports:

```text
artifacts/pima_best_pipeline.joblib
```

The bundle includes:

- calibrated pipeline
- feature list
- selected operating threshold
- CV summary
- tuning summary
- validation threshold summary
- held-out test metrics
- run metadata

See [`artifacts/README.md`](artifacts/README.md) for details.

---

## 🔮 Score a CSV

After running the notebook and exporting the artifact:

```bash
python scripts/predict.py \
  --csv data/raw/diabetes.csv \
  --out artifacts/scored.csv
```

By default, the script loads:

```text
artifacts/pima_best_pipeline.joblib
```

To use a different exported artifact, pass `--artifact` explicitly:

```bash
python scripts/predict.py \
  --csv data/raw/diabetes.csv \
  --artifact artifacts/pima_best_pipeline.joblib \
  --out artifacts/scored.csv
```

The output CSV adds:

| Column | Meaning |
|---|---|
| `diabetes_proba` | calibrated diabetes-positive probability |
| `diabetes_pred` | thresholded prediction using the exported operating threshold |

---

## 📌 Notes

- `imbalanced-learn` is optional at runtime inside the notebook. If it is unavailable or incompatible, the notebook falls back to estimator-side imbalance handling.
- `xgboost` is optional in the notebook. If unavailable, the XGBoost candidate is skipped.
- Cost values are illustrative and should be aligned with the intended decision policy before any operational use.

---

## 📄 License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
