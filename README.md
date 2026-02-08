# 🩺 Pima Diabetes Prediction — Cost-Aware Pipeline

A compact, production-minded workflow to predict diabetes risk from routine clinical measurements with
probability calibration and an explicit threshold policy.

Case study: `CASE_STUDY.md`

---

## What this repository includes
- Main notebook: `diabetes-prediction-from-eda-to-production.ipynb`
- Optional reference notebook: `pima-indians-diabetes-database.ipynb`
- Exported artifacts under `./artifacts/`
- Optional scoring script: `scripts/predict.py`

---

## Dataset
Source: Kaggle “Pima Indians Diabetes Database” (`diabetes.csv`).

Expected columns (typical):
- `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`

### Local (recommended)
1) Download the dataset CSV.
2) Place it at:
`data/raw/diabetes.csv`

### Kaggle
The notebook also supports:
`/kaggle/input/pima-indians-diabetes-database/diabetes.csv`

---

## Getting started

### 1) Install
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the notebook
Open and run:
- `diabetes-prediction-from-eda-to-production.ipynb`

The notebook will:
- clean and audit the dataset
- train models and calibrate probabilities
- select an operating threshold (policy)
- export `artifacts/pima_best_pipeline.joblib`

---

## Artifacts
The exported bundle includes:
- trained pipeline
- operating threshold
- run metadata

See `artifacts/README.md`.

---

## Score a CSV (optional)
After exporting artifacts, you can score a CSV:

```bash
python scripts/predict.py --csv data/raw/diabetes.csv --out artifacts/scored.csv
```

The output adds:
- `diabetes_proba`
- `diabetes_pred`

---

## Methodology notes
- Probability calibration makes thresholds usable for decisions.
- Threshold policy is selected on validation and exported with the artifact.

---

## License
MIT (code). Dataset licensing depends on the dataset source where you download it.
