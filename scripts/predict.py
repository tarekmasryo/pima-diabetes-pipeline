#!/usr/bin/env python
import argparse
from pathlib import Path

import joblib
import pandas as pd


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}. Run the notebook to generate artifacts.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid artifact format. Expected a dict with a 'pipeline' key.")
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Score Pima diabetes samples using exported artifacts.")
    p.add_argument("--csv", required=True, help="Path to a CSV containing samples to score.")
    p.add_argument("--artifact", default="artifacts/pima_best_pipeline.joblib", help="Path to exported artifact bundle.")
    p.add_argument("--out", default="artifacts/scored.csv", help="Output CSV path.")
    args = p.parse_args()

    bundle = load_bundle(Path(args.artifact))
    pipe = bundle["pipeline"]
    features = bundle.get("features")

    df = pd.read_csv(args.csv)
    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        X = df[features].copy()
    else:
        X = df.copy()

    # Prefer calibrated probabilities if available
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        # Fallback: decision function → pseudo probability not available
        raise RuntimeError("Pipeline does not support predict_proba. Re-run notebook with calibration enabled.")

    thr = float(bundle.get("operating_threshold", 0.5))
    pred = (proba >= thr).astype(int)

    out = df.copy()
    out["diabetes_proba"] = proba
    out["diabetes_pred"] = pred

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path} (threshold={thr:.3f})")


if __name__ == "__main__":
    main()
