#!/usr/bin/env python
"""Score Pima diabetes CSV files with the exported notebook artifact.

The notebook can export a joblib bundle that contains sklearn transformers using
functions/classes defined in the notebook. They are mirrored here so joblib can
load the artifact when this script is executed as __main__.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def zeros_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Replace clinically impossible zero values with NaN inside the pipeline."""
    X = X.copy()
    X[ZERO_AS_MISSING_COLUMNS] = X[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return X


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features after zero handling and before imputation."""
    X = X.copy()
    X["Age_BMI"] = X["Age"] * X["BMI"]
    X["Glucose_BMI"] = X["Glucose"] * X["BMI"]
    X["Preg_Age_Ratio"] = X["Pregnancies"] / (X["Age"] + 1)
    return X


class SafeSklearnPipeline(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-compatible pipeline used when imblearn is unavailable."""

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        return dict(getattr(self, "steps_", self.steps))

    def get_params(self, deep=True):
        params = {"steps": self.steps}
        if not deep:
            return params

        for name, step in self.steps:
            params[name] = step
            if hasattr(step, "get_params"):
                for key, value in step.get_params(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        if "steps" in params:
            self.steps = params.pop("steps")

        step_lookup = dict(self.steps)
        for key, value in params.items():
            if "__" in key:
                step_name, sub_key = key.split("__", 1)
                if step_name not in step_lookup:
                    raise ValueError(f"Unknown pipeline step: {step_name}")
                step_lookup[step_name].set_params(**{sub_key: value})
            else:
                updated = False
                new_steps = []
                for name, step in self.steps:
                    if name == key:
                        new_steps.append((name, value))
                        updated = True
                    else:
                        new_steps.append((name, step))
                if not updated:
                    raise ValueError(f"Unknown pipeline parameter: {key}")
                self.steps = new_steps
                step_lookup = dict(self.steps)
        return self

    def fit(self, X, y=None, **fit_params):
        self.steps_ = [(name, clone(step)) for name, step in self.steps]
        Xt = X
        for _, step in self.steps_[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                Xt = step.fit(Xt, y).transform(Xt)

        _, final_estimator = self.steps_[-1]
        final_estimator.fit(Xt, y, **fit_params)
        self.classes_ = getattr(final_estimator, "classes_", None)
        self.n_features_in_ = getattr(final_estimator, "n_features_in_", None)
        return self

    def _transform(self, X):
        Xt = X
        for _, step in self.steps_[:-1]:
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X):
        return self.steps_[-1][1].predict(self._transform(X))

    def predict_proba(self, X):
        return self.steps_[-1][1].predict_proba(self._transform(X))

    def score(self, X, y):
        return self.steps_[-1][1].score(self._transform(X), y)


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}. Run the notebook to generate artifacts.")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid artifact format. Expected a dict with a 'pipeline' key.")
    return bundle


def score_csv(csv_path: Path, artifact_path: Path, out_path: Path) -> Path:
    bundle = load_bundle(artifact_path)
    pipeline = bundle["pipeline"]
    features = bundle.get("features")

    df = pd.read_csv(csv_path)
    if features:
        missing = [column for column in features if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        X = df[features].copy()
    else:
        X = df.copy()

    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("Pipeline does not support predict_proba. Re-run the notebook with calibration enabled.")

    proba = pipeline.predict_proba(X)[:, 1]
    threshold = float(bundle.get("operating_threshold", 0.5))
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["diabetes_proba"] = proba
    out["diabetes_pred"] = pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Pima diabetes samples using exported artifacts.")
    parser.add_argument("--csv", required=True, help="Path to a CSV containing samples to score.")
    parser.add_argument(
        "--artifact",
        default="artifacts/pima_best_pipeline.joblib",
        help="Path to the exported artifact bundle.",
    )
    parser.add_argument("--out", default="artifacts/scored.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = score_csv(Path(args.csv), Path(args.artifact), Path(args.out))
    bundle = load_bundle(Path(args.artifact))
    threshold = float(bundle.get("operating_threshold", 0.5))
    print(f"Saved -> {out_path} (threshold={threshold:.3f})")


if __name__ == "__main__":
    main()
