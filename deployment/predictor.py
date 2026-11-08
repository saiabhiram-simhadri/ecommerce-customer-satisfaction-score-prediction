"""Reproduce the notebook's 142-feature, unscaled ANN input contract."""
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "csat_deployment_artifacts"
CATEGORICAL = ["channel_name", "category", "product_category", "tenure_bucket", "agent_shift"]
NUMERIC = ["item_price", "connected_handling_time", "response_time_hours",
           "order_to_issue_days", "survey_lag_hours", "issue_hour", "is_weekend"]
INPUT_COLUMNS = CATEGORICAL + ["customer_remarks"] + NUMERIC


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Predictor:
    def __init__(self, artifact_dir=ARTIFACTS):
        import tensorflow as tf
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "deployment_metadata.json").read_text())
        self.vectorizer = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
        self.model = tf.keras.models.load_model(artifact_dir / "champion_csat_ann.keras", compile=False)
        if self.model.input_shape[-1] != len(self.metadata["input_features"]):
            raise ValueError("Model and input schema do not match. Re-export the deployment bundle.")

    def validate(self, records):
        """Check inputs and fill empty values with the exported defaults."""
        if not isinstance(records, pd.DataFrame) or records.empty:
            raise ValueError("Provide at least one interaction.")
        if len(records) > 10000:
            raise ValueError("Use at most 10,000 interactions per batch.")
        missing = sorted(set(INPUT_COLUMNS) - set(records.columns))
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        data = records.loc[:, INPUT_COLUMNS].copy().reset_index(drop=True)
        for col in CATEGORICAL:
            data[col] = data[col].astype("string").str.strip()
            invalid = ~data[col].isin(self.metadata["categories"][col])
            if invalid.any():
                raise ValueError(f"Unsupported {col}. Use the categories shown in the app.")
        data["customer_remarks"] = data["customer_remarks"].fillna("No remarks provided").astype(str)
        data.loc[data["customer_remarks"].str.strip().eq(""), "customer_remarks"] = "No remarks provided"
        if data["customer_remarks"].str.len().gt(20000).any():
            raise ValueError("Customer remarks must be at most 20,000 characters each.")
        for col in NUMERIC:
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except (TypeError, ValueError):
                raise ValueError(f"{col} must contain numbers or empty cells.")
            data[col] = data[col].fillna(self.metadata["defaults"][col])
            if not np.isfinite(data[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{col} must contain finite numbers.")
            if data[col].lt(0).any():
                raise ValueError(f"{col} cannot be negative.")
        if (~data["issue_hour"].between(0, 23) | data["issue_hour"].mod(1).ne(0)).any():
            raise ValueError("issue_hour must be an integer from 0 to 23.")
        if (~data["is_weekend"].isin([0, 1])).any():
            raise ValueError("is_weekend must be 0 or 1.")
        if data["survey_lag_hours"].ne(0).any():
            raise ValueError("survey_lag_hours must be 0: this feature was constant in the notebook's training data.")
        return data

    def transform(self, records):
        """Build the same numeric features, text vectors, and categories as training."""
        data = self.validate(records)
        for col, upper in self.metadata["numeric_caps"].items():
            data[col] = data[col].clip(0, upper)
        # 1. Numerical features and text counts.
        cleaned = data["customer_remarks"].map(clean_text)
        out = pd.DataFrame(index=data.index)
        for col in ["order_to_issue_days", "survey_lag_hours", "is_weekend"]:
            out[col] = data[col]
        out["remark_char_length"] = data["customer_remarks"].str.len()
        out["remark_word_count"] = data["customer_remarks"].str.split().str.len()
        keywords = "|".join(self.metadata["negative_triggers"])
        pattern = rf"\b(?:{keywords})\b"
        out["has_negative_keywords"] = cleaned.str.contains(pattern).astype(int)
        out["issue_hour_sin"] = np.sin(2 * np.pi * data["issue_hour"] / 24)
        out["issue_hour_cos"] = np.cos(2 * np.pi * data["issue_hour"] / 24)
        out["handling_to_price_ratio"] = (data["connected_handling_time"] + 1) / np.log1p(data["item_price"] + 1)
        for col in ["item_price", "connected_handling_time", "response_time_hours"]:
            out[f"log_{col}"] = np.log1p(data[col])
        # 2. The 100 text features, using the already-fitted vectorizer.
        text = pd.DataFrame(self.vectorizer.transform(cleaned).toarray(),
                            columns=[f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()])
        # 3. Category indicators. Fix category order to match the notebook.
        for col in CATEGORICAL:
            data[col] = pd.Categorical(data[col], categories=self.metadata["categories"][col])
        categories = pd.get_dummies(data[CATEGORICAL], drop_first=True, dtype=float)
        out = pd.concat([out, text, categories], axis=1)
        out = out[self.metadata["input_features"]]
        if not np.isfinite(out.to_numpy(dtype=float)).all():
            raise ValueError("Preprocessing produced missing or non-finite model features.")
        # The notebook trained on these features directly. Never apply robust_scaler.joblib.
        return out.astype(float)

    def predict(self, records):
        """Convert inputs to features, run the ANN, and return ratings and probabilities."""
        features = self.transform(records)
        model_inputs = features.to_numpy(dtype=np.float32)
        probabilities = np.asarray(self.model(model_inputs, training=False))
        if (probabilities.shape != (len(features), 5) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("The model returned invalid class probabilities.")
        result = pd.DataFrame({"predicted_csat": probabilities.argmax(axis=1) + 1,
                               "top_probability": probabilities.max(axis=1),
                               "low_csat_probability": probabilities[:, :2].sum(axis=1)})
        result["follow_up"] = result["predicted_csat"].le(2)
        for idx in range(5):
            result[f"probability_csat_{idx + 1}"] = probabilities[:, idx]
        return result
