"""Export existing notebook state; do not retrain or edit the source notebook."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np

from deployment.predictor import ARTIFACTS, CATEGORICAL, INPUT_COLUMNS, ROOT, Predictor


def export_from_session(namespace):
    required = ["champion_ann", "tfidf", "X_train", "X_test", "df", "eda_df", "negative_triggers"]
    missing = [key for key in required if key not in namespace]
    if missing:
        raise ValueError(f"Run the notebook through ANN selection first. Missing: {missing}")
    frame, observed = namespace["df"], namespace["eda_df"]
    ARTIFACTS.mkdir(exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_notebook_sha256": hashlib.sha256((ROOT / "E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb").read_bytes()).hexdigest(),
        "input_features": namespace["X_train"].columns.tolist(),
        "categories": {c: sorted(frame[c].astype(str).unique().tolist()) for c in CATEGORICAL},
        "defaults": {c: float(frame[c].median()) for c in ["item_price", "connected_handling_time", "response_time_hours", "order_to_issue_days", "survey_lag_hours"]},
        "negative_triggers": namespace["negative_triggers"],
        "numeric_caps": {},
        "scaling": "none: notebook ANN uses unscaled engineered features",
        "model_version": datetime.now(timezone.utc).strftime("notebook-ann-%Y%m%dT%H%M%SZ"),
        "class_mapping": {str(i): i + 1 for i in range(5)},
        "training_rows": len(namespace["X_train"]),
        "packages": {p: importlib.metadata.version(p) for p in ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "joblib"]},
        "limitations": ["Retrospective rated-interaction prototype; early-warning performance is unverified.",
                        "Preprocessing and text vocabulary were fitted before splitting in the source notebook.",
                        "Survey lag was constant zero in training; the app requires zero.",
                        "Price and handling-time caps collapse most values to 979 and 427 respectively.",
                        "Probabilities are uncalibrated. Historical CSAT is required to measure accuracy."]
    }
    metadata["defaults"].update(issue_hour=12, is_weekend=0)
    for col in ["item_price", "connected_handling_time"]:
        filled = observed[col].fillna(observed[col].median())
        q25, q75 = filled.quantile([.25, .75])
        metadata["numeric_caps"][col] = float(q75 + 2.5 * (q75 - q25))
        np.testing.assert_allclose(filled.clip(0, metadata["numeric_caps"][col]), frame[col])
    namespace["champion_ann"].save(ARTIFACTS / "champion_csat_ann.keras")
    joblib.dump(namespace["tfidf"], ARTIFACTS / "tfidf_vectorizer.joblib")
    (ARTIFACTS / "deployment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    predictor = Predictor()
    # Compare all held-out feature rows, including every observed category and text vector.
    indices = namespace["X_test"].index
    expected = namespace["X_test"]
    max_difference = 0.0
    for start in range(0, len(indices), 1000):
        index = indices[start:start + 1000]
        inputs = frame.loc[index, INPUT_COLUMNS]
        actual = predictor.transform(inputs).to_numpy()
        original = expected.loc[index].to_numpy(dtype=float)
        np.testing.assert_allclose(actual, original, rtol=1e-7, atol=1e-7)
        max_difference = max(max_difference, float(np.max(np.abs(actual - original))))
    sample = expected.iloc[:128].to_numpy(dtype=np.float32)
    before = np.asarray(namespace["champion_ann"](sample, training=False))
    after = np.asarray(predictor.model(sample, training=False))
    np.testing.assert_allclose(before, after, atol=1e-6, rtol=1e-6)
    report = {"held_out_rows_checked": len(indices), "features_checked": len(expected.columns),
              "maximum_feature_difference": max_difference,
              "prediction_rows_checked": len(sample),
              "maximum_probability_difference": float(np.max(np.abs(before - after))),
              "source_notebook_sha256": metadata["source_notebook_sha256"]}
    (ARTIFACTS / "parity_report.json").write_text(json.dumps(report, indent=2))
    # Store a numerical regression fixture, without customer remarks or identifiers.
    np.savez_compressed(ARTIFACTS / "parity_fixture.npz", features=sample, probabilities=before)
    return report
