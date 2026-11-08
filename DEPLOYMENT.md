# Local CSAT demonstration

The Streamlit app loads the existing notebook ANN; it does not train a replacement model or change the source notebook. Artifacts are exported from the active notebook session. All normal app use works offline after dependencies and artifacts are installed.

## Start on this computer

From PowerShell in this project:

```powershell
.\run_demo.ps1
```

Open **http://127.0.0.1:8501**. Stop the server with Ctrl+C. If PowerShell blocks script execution, run the command directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

The server binds to localhost. No cloud account or hosted site is required.

## Video walkthrough (about 3 minutes)

1. Show the notebook's validation comparison and explain that this is a retrospective prototype for rated interactions.
2. Open **Single prediction**, select **Positive support**, and click **Predict CSAT**. Explain the 1–5 rating, five probabilities, and follow-up policy (predicted rating 1–2).
3. Select **Delayed refund**, predict again, then change the remarks or response delay and resubmit. The model's outputs may change; do not promise a specific rating or a monotonic response.
4. Open **Batch predictions**. Download the synthetic CSV template. Upload it, uncheck **Use the bundled synthetic demo batch**, and click **Run batch predictions**. Alternatively, leave the demo checkbox selected for an offline fallback. Download the predictions.
5. Open **History** to show accumulated predictions, predicted rating distribution, and saved results. Restarting the app preserves history. Editing inputs or refreshing does not retrain the model.
6. Explain that these charts describe model predictions. Measuring live accuracy requires real CSAT labels collected later.

Synthetic examples are demonstrations, not labeled evaluation data. Actual predictions are produced by the exported ANN.

## Input contract

The code follows a short flow: `app.py` collects inputs, `deployment/predictor.py` validates and transforms them before calling the ANN, and `deployment/history.py` saves and reads prediction summaries. `deployment/export.py` is only needed when exporting a trained model again.

Use `demo_interactions.csv` as the batch template. The app accepts measured elapsed durations, **not original dataset timestamps**. It ignores extra columns, including an optional actual CSAT label, rather than using them as predictors.

| Column | Meaning |
|---|---|
| `channel_name`, `category`, `product_category`, `tenure_bucket`, `agent_shift` | Exact categories displayed in the form. Unknown categories are rejected. |
| `customer_remarks` | Text available at scoring time; empty entries become `No remarks provided`. |
| `item_price` | Price in source dataset units; capped at 979. |
| `connected_handling_time` | Handling time in source dataset units; capped at 427. The notebook does not establish the time unit. |
| `response_time_hours` | Hours from issue report to response. |
| `order_to_issue_days` | Days from order to issue report. |
| `survey_lag_hours` | Must be 0 because the training feature is constant zero. |
| `issue_hour` | Integer hour 0–23 in the source interaction's local time. |
| `is_weekend` | 1 for Saturday/Sunday, otherwise 0. |

All columns must exist. Blank numeric cells use the exported notebook defaults, including noon for missing hour and weekday for missing weekend status. Negative or non-finite numbers and malformed values are rejected. Limits: 10 MB CSV, 10,000 rows, 20,000 characters per remark. Predictions are logged only on explicit submissions.

## Artifacts and verification

`csat_deployment_artifacts/` contains the Keras model, fitted TF-IDF vectorizer, deployment metadata, a numerical regression fixture, and `parity_report.json`. Only load trusted model/joblib artifacts.

The deployment reproduces text cleaning, raw-text counts, keyword flags, caps, log transforms, cyclical hour features, handling/price ratio, dropped-first category encoding, and exact model feature order. **No robust scaler is applied:** the notebook trained the ANN without it. A separately exported `robust_scaler.joblib` is intentionally unused.

Export checks reconstructed features against **all 12,887 held-out rows and 142 features**, and compares reloaded model probabilities against the notebook model on 128 rows. The fixture contains numeric model inputs and probabilities, not customer remarks or identifiers.

Run the integration checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Checks cover model serialization parity, preprocessing, categories, invalid inputs, batch predictions, SQLite history, and Streamlit form submissions/reruns using Streamlit AppTest. They use a temporary history database.

## Export again after changing the notebook model

Run the existing notebook through champion ANN selection. In that notebook's Python console (or a temporary cell if you choose), run:

```python
from deployment.export import export_from_session
export_from_session(globals())
```

The exporter reads trained objects from memory, checks parity, and writes separate deployment files. It never writes the notebook. Restart Streamlit after replacing artifacts to clear the cached model.

## Environment and history

Installed dependencies are pinned in `requirements-deployment.txt`. For a fresh Python 3.12 virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-deployment.txt
```

Prediction summaries persist in `local_data/predictions.sqlite3` (ignored by Git). Remarks and raw input details are not persisted. History includes UTC timestamp, model version, channel/category, predicted rating, top probability, low-score probability, and follow-up flag. Repeated submissions count as new interactions. `CSAT_HISTORY_PATH` can select another database, for example an isolated rehearsal history.

## Model limitations to mention

The notebook reports validation accuracy **40.14%**, macro F1 **0.2588**, and score-1 recall **42.73%**. These figures are recorded notebook results, not newly measured deployment performance. Rare ratings remain weak, class imbalance is substantial, and probabilities have not been calibrated. Some preprocessing was fitted before splitting; feedback may be available only alongside the actual survey. The original caps remove much of price and handling-time variation. This deployment preserves those choices for parity. Validating early-warning use and improving preprocessing require a separate model revision and evaluation.
