import json
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from deployment.history import read_history, save_predictions
from deployment.predictor import ARTIFACTS, CATEGORICAL, ROOT, Predictor


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()
        cls.demo = pd.read_csv(ROOT / "demo_interactions.csv")

    def test_export_matches_notebook(self):
        report = json.loads((ARTIFACTS / "parity_report.json").read_text())
        self.assertEqual(report["held_out_rows_checked"], 12887)
        self.assertEqual(report["features_checked"], 142)
        self.assertLess(report["maximum_feature_difference"], 1e-7)
        fixture = np.load(ARTIFACTS / "parity_fixture.npz")
        current = np.asarray(self.predictor.model(fixture["features"], training=False))
        np.testing.assert_allclose(current, fixture["probabilities"], atol=1e-6, rtol=1e-6)

    def test_batch_predictions_and_feature_order(self):
        features = self.predictor.transform(self.demo)
        self.assertEqual(features.columns.tolist(), self.predictor.metadata["input_features"])
        results = self.predictor.predict(self.demo)
        self.assertEqual(len(results), len(self.demo))
        self.assertTrue(results.predicted_csat.between(1, 5).all())
        np.testing.assert_allclose(results.filter(like="probability_csat_").sum(axis=1), 1, atol=1e-6)
        self.assertTrue((results.follow_up == results.predicted_csat.le(2)).all())
        shuffled = self.demo.iloc[[2, 0, 1]]
        np.testing.assert_allclose(self.predictor.predict(shuffled).filter(like="probability_csat_"),
                                   results.iloc[[2, 0, 1]].filter(like="probability_csat_"), atol=1e-6)

    def test_categories_match_drop_first_encoding(self):
        for col in CATEGORICAL:
            for index, category in enumerate(self.predictor.metadata["categories"][col]):
                record = self.demo.iloc[:1].copy()
                record[col] = category
                features = self.predictor.transform(record)
                columns = [c for c in features if c.startswith(col + "_")]
                self.assertEqual(int(features[columns].sum(axis=1).iloc[0]), int(index > 0))

    def test_caps_missing_values_and_text(self):
        record = self.demo.iloc[:1].copy()
        record["item_price"] = 100000.0
        record["connected_handling_time"] = 100000.0
        record["customer_remarks"] = "REFUND!!! Poor service."
        features = self.predictor.transform(record)
        self.assertAlmostEqual(features.log_item_price.iloc[0], np.log1p(979))
        self.assertAlmostEqual(features.log_connected_handling_time.iloc[0], np.log1p(427))
        self.assertEqual(features.has_negative_keywords.iloc[0], 1)
        self.assertEqual(features.remark_char_length.iloc[0], len("REFUND!!! Poor service."))
        record["customer_remarks"] = None
        record["item_price"] = np.nan
        self.assertTrue(np.isfinite(self.predictor.transform(record).to_numpy()).all())

    def test_reject_invalid_inputs(self):
        cases = [("response_time_hours", -1), ("item_price", float("inf")),
                 ("item_price", "wrong"), ("issue_hour", 24), ("issue_hour", 1.5),
                 ("is_weekend", 2), ("channel_name", "Unknown"), ("survey_lag_hours", 1)]
        for column, value in cases:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                record = self.demo.iloc[:1].copy().astype(object)
                record[column] = value
                self.predictor.predict(record)
        for frame in [self.demo.iloc[:0], self.demo.drop(columns="category")]:
            with self.assertRaises(ValueError):
                self.predictor.predict(frame)

    def test_history_persistence_without_remarks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.sqlite3"
            self.assertTrue(read_history(path).empty)
            result = self.predictor.predict(self.demo)
            save_predictions(self.demo, result, "test", "batch", path)
            history = read_history(path)
            self.assertEqual(len(history), len(self.demo))
            self.assertNotIn("customer_remarks", history)
            self.assertEqual(len(read_history(path)), len(self.demo))

    def test_streamlit_interactions_and_reruns(self):
        with tempfile.TemporaryDirectory() as folder:
            old_path = os.environ.get("CSAT_HISTORY_PATH")
            path = str(Path(folder) / "app.sqlite3")
            os.environ["CSAT_HISTORY_PATH"] = path
            try:
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(app.session_state["workspace"], "Single prediction")
                next(b for b in app.button if b.label == "Predict CSAT").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(m.label == "Predicted CSAT" for m in app.metric))
                self.assertEqual(len(read_history(path)), 1)
                app.run()
                self.assertEqual(len(read_history(path)), 1, "Reruns must not duplicate predictions")
                app.selectbox[0].select("Delayed refund").run()
                next(b for b in app.button if b.label == "Predict CSAT").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(read_history(path)), 2)
                app.sidebar.button(key="nav_Batch predictions").click().run()
                next(b for b in app.button if b.label == "Run batch predictions").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(len(read_history(path)), 2 + len(self.demo))
                # Exercise uploaded CSV handling with the real uploader return contract.
                uploaded = BytesIO((ROOT / "demo_interactions.csv").read_bytes())
                uploaded.size = len(uploaded.getvalue())
                next(c for c in app.checkbox if c.label == "Use the bundled synthetic demo batch").uncheck()
                with patch("streamlit.file_uploader", return_value=uploaded):
                    next(b for b in app.button if b.label == "Run batch predictions").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(len(read_history(path)), 2 + 2 * len(self.demo))
                invalid = BytesIO(b"wrong_column\ninvalid\n")
                invalid.size = len(invalid.getvalue())
                with patch("streamlit.file_uploader", return_value=invalid):
                    next(b for b in app.button if b.label == "Run batch predictions").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Missing columns" in error.value for error in app.error))
                self.assertEqual(len(read_history(path)), 2 + 2 * len(self.demo))
                self.assertTrue(any("Scored 6" in s.value for s in app.success))
                app.sidebar.button(key="nav_History").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(m.label == "Interactions scored" for m in app.metric))
                next(b for b in app.button if b.label == "Refresh dashboard").click().run()
                self.assertEqual(len(read_history(path)), 2 + 2 * len(self.demo))
            finally:
                if old_path is None:
                    os.environ.pop("CSAT_HISTORY_PATH", None)
                else:
                    os.environ["CSAT_HISTORY_PATH"] = old_path


if __name__ == "__main__":
    unittest.main()
