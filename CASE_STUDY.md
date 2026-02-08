# Case Study — Pima Diabetes Pipeline (Decision-Grade Classification)

## Overview
This repository provides a compact, production-minded workflow to predict diabetes risk from routine clinical measurements:
EDA → preprocessing → model training → probability calibration → threshold policy → exportable artifacts.

The goal is not just a good score, but a decision policy that controls false positives vs false negatives.

## The real problem
Medical risk screening is asymmetric:
- **False negatives** miss high-risk cases.
- **False positives** increase follow-up cost and patient friction.

A model score is not enough; you need calibrated probabilities and an explicit threshold policy.

## Goals (definition of done)
**Functional goals**
- Clean the dataset (handle impossible zeros and missingness).
- Train a strong baseline and a stronger model.
- Calibrate probabilities to make thresholds meaningful.
- Choose an operating threshold with a clear policy (default: cost-aware).

**Engineering goals**
- Reproducible analysis in a single notebook.
- Export a reusable artifact bundle for inference.

## Approach
### 1) Data integrity and preprocessing
Several medical columns contain impossible zeros (e.g., glucose, BMI). The workflow:
- converts impossible zeros to missing values
- imputes missing values with robust statistics
- keeps the target definition explicit (`Outcome`)

### 2) Modeling and calibration
The workflow trains a baseline and a stronger model, then applies probability calibration so scores behave like probabilities.
This makes threshold selection defensible.

### 3) Decision policy (thresholding)
Instead of using a default 0.5 threshold, the workflow selects an operating point.
By default, it uses a simple cost curve:
- higher cost for FN vs FP (configurable in the notebook)

The selected threshold is exported with the artifact bundle so inference is consistent.

## Outputs
The notebook exports a bundle to `./artifacts/pima_best_pipeline.joblib` containing:
- the trained (calibrated) pipeline
- the feature list used for scoring
- the selected operating threshold
- run metadata (cost settings, model name, CV summary)

## Usage
Execution steps are documented in `README.md`.

## Limitations
- This is a single-dataset workflow; generalization requires external validation.
- Costs are illustrative; thresholds should be re-selected based on real operational costs.
- Demographic/clinical shift over time can affect performance.

## Next steps
- Add external validation or a temporal split if new data is available.
- Evaluate cost sensitivity (how decisions change with FP/FN costs).
- Add simple monitoring hooks (drift checks on key features).
