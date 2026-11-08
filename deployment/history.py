"""Save and read prediction summaries in a local SQLite database."""
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import uuid

import pandas as pd

from deployment.predictor import ROOT

DEFAULT_DB = ROOT / "local_data" / "predictions.sqlite3"


def open_database(path=None):
    path = Path(path or os.environ.get("CSAT_HISTORY_PATH", DEFAULT_DB))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY, created_at TEXT, model_version TEXT,
        source TEXT, channel_name TEXT, category TEXT, predicted_csat INTEGER,
        top_probability REAL, low_csat_probability REAL, follow_up INTEGER)""")
    return connection


def save_predictions(inputs, results, version, source, path=None):
    """Save one summary per prediction, leave customer remarks out of history."""
    history = results[["predicted_csat", "top_probability", "low_csat_probability", "follow_up"]].copy()
    history["prediction_id"] = [str(uuid.uuid4()) for _ in range(len(results))]
    history["created_at"] = datetime.now(timezone.utc).isoformat()
    history["model_version"] = version
    history["source"] = source
    history["channel_name"] = inputs["channel_name"].to_numpy()
    history["category"] = inputs["category"].to_numpy()
    with closing(open_database(path)) as connection:
        history.to_sql("predictions", connection, if_exists="append", index=False)


def read_history(path=None):
    """Read saved predictions, newest first."""
    with closing(open_database(path)) as connection:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC", connection)
