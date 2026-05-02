# Case Study — Pima Diabetes Leakage-Safe ML Pipeline

## Overview

This repository demonstrates a compact, decision-aware machine-learning workflow for the Pima Diabetes benchmark:

```text
EDA → leakage-safe preprocessing → model comparison → calibration → threshold policy → held-out test evaluation → exportable artifact
```

The goal is not just to report a score. The goal is to show a reproducible workflow where preprocessing, evaluation, threshold selection, and artifact export are handled in a way that is easy to inspect and rerun.

## Problem framing

Binary medical-risk benchmarks are asymmetric by nature:

- false negatives may miss diabetes-positive records
- false positives increase follow-up burden

For that reason, the notebook evaluates ranking quality, calibration, and threshold behavior instead of relying on accuracy alone.

## Technical approach

### 1) Data integrity

The notebook validates the expected Pima schema and inspects medically impossible zero values in columns such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.

### 2) Leakage-safe preprocessing

Zero-to-missing conversion, interaction features, median imputation, scaling, optional imbalance handling, and model fitting are kept inside the modeling pipeline. This prevents preprocessing statistics from leaking across validation or test boundaries.

### 3) Model comparison and tuning

The workflow compares baseline and stronger model candidates, then tunes selected models using inner cross-validation.

### 4) Probability calibration

The selected model is calibrated so that threshold analysis is based on probability-like scores rather than raw classifier outputs.

### 5) Threshold policy

The operating threshold is selected on the validation split only using an illustrative false-positive / false-negative cost setup. The held-out test set is used once for final reporting.

## Outputs

The notebook exports `artifacts/pima_best_pipeline.joblib`, a bundle containing:

- calibrated pipeline
- selected operating threshold
- feature list
- model name
- CV and tuning summaries
- validation threshold summary
- held-out test metrics
- run metadata

## Limitations

- This is a benchmark workflow, not a clinical model.
- Real-world use would require external validation on representative data.
- The illustrative cost policy should be replaced with domain-approved decision costs before operational use.
- Monitoring for calibration drift and population drift would be required in an operational setting.

## Recommended next steps

- Evaluate the workflow on additional representative data.
- Test sensitivity to different threshold policies.
- Add lightweight monitoring checks around input distributions, calibration, and output rates.
