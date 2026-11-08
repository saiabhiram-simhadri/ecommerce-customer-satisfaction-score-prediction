# Deployment and Usage Guide

This guide explains how to install, run, and maintain the CSAT prediction application from a fresh clone. The Streamlit app uses the bundled ANN and preprocessing artifacts, training is not required to start the application.

For the project overview and validation results, see [README.md](README.md). The analysis and training workflow is in the [notebook](E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb).

## Prerequisites

- Git.
- Python **3.12**, the version used for the documented deployment environment.
- Internet access to clone the repository and install dependencies.
- A writable location for the local SQLite history database.

Application dependencies are pinned in [requirements-deployment.txt](requirements-deployment.txt). Once installed, normal application use works offline with the bundled artifacts and synthetic examples. The source training dataset is not needed for inference.

## 1. Clone the repository

```shell
git clone https://github.com/saiabhiram-simhadri/ecommerce-customer-satisfaction-score-prediction.git
cd ecommerce-customer-satisfaction-score-prediction
```

Run the following commands from the repository root.

## 2. Create an environment and install dependencies

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deployment.txt
```

### Linux or macOS

With Python 3.12 installed:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements-deployment.txt
```

The commands use the virtual environment's interpreter directly, so environment activation is optional. Availability of the pinned TensorFlow and other package builds depends on the operating system and CPU architecture.

## 3. Launch the application

### Windows PowerShell

```powershell
.\run_demo.ps1
```

If PowerShell blocks script execution, launch Streamlit directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

### Linux or macOS

```bash
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501** in a browser. Stop the server with **Ctrl+C** in the terminal.

The supplied configuration binds the application to localhost. This guide covers local operation, the repository does not configure a public hosting service. Server limits and appearance are configured in [.streamlit/config.toml](.streamlit/config.toml).

## 4. Use the application

### Single prediction

1. Open **Single prediction**.
2. Select a synthetic example or edit its support details, timing values, and customer remarks.
3. Click **Predict CSAT**.
4. Review the predicted rating, class probabilities, and follow-up flag.

Predicted ratings **1–2** are flagged for follow-up. This is a demonstration rule. Editing form values requires another submission to generate a new prediction.

### Batch predictions

1. Open **Batch predictions**.
2. Download the CSV template or use [demo_interactions.csv](demo_interactions.csv).
3. Upload a CSV and clear **Use the bundled synthetic demo batch** to process the uploaded records. Keep it selected to use the bundled examples.
4. Click **Run batch predictions** and download the results.

Synthetic examples demonstrate the workflow and are not labeled evaluation data.

### History

Open **History** to review saved prediction summaries, rating distributions, and follow-up counts, or download the history as CSV. Records persist across application restarts. Each submitted prediction creates a new history entry, including repeated submissions of the same interaction.

History charts describe predictions, rather than actual survey outcomes or measured accuracy.

## CSV input requirements

The application accepts elapsed durations rather than raw dataset timestamps. Every required column must be present. Extra columns, including an actual CSAT label, are ignored as predictors.

| Column | Accepted values and meaning |
| --- | --- |
| `channel_name` | Support channel, use an exact category shown in the form. |
| `category` | Issue category, use an exact category shown in the form. |
| `product_category` | Product group, use an exact category shown in the form. |
| `tenure_bucket` | Agent tenure group, use an exact category shown in the form. |
| `agent_shift` | Agent shift, use an exact category shown in the form. |
| `customer_remarks` | Text available when scoring. Blank remarks become `No remarks provided`. |
| `item_price` | Nonnegative price in dataset units, capped at 979 before inference. |
| `connected_handling_time` | Nonnegative handling time in dataset units, capped at 427. The source notebook does not establish its time unit. |
| `response_time_hours` | Nonnegative hours from issue report to response. |
| `order_to_issue_days` | Nonnegative days from order to issue report. |
| `survey_lag_hours` | Must be 0, this feature was constant zero during training. |
| `issue_hour` | Integer from 0 to 23 in the interaction's local time. |
| `is_weekend` | 1 for Saturday or Sunday, otherwise 0. |

Unknown or blank categories are rejected. Supported category values are listed in [deployment_metadata.json](csat_deployment_artifacts/deployment_metadata.json) under `categories` and displayed in the single-prediction form.

Blank numeric cells use the exported defaults:

| Column | Default |
| --- | ---: |
| `item_price` | 979 |
| `connected_handling_time` | 427 |
| `response_time_hours` | 0.08333333333333333 |
| `order_to_issue_days` | 9.039583333333333 |
| `survey_lag_hours` | 0 |
| `issue_hour` | 12 |
| `is_weekend` | 0 |

Negative numbers, non-finite values, and malformed numeric values are rejected. Upload limits are **10 MB per CSV**, **10,000 interactions per batch**, and **20,000 characters per remark**.

## Prediction storage

The default history database is `local_data/predictions.sqlite3`. Its parent directory is created automatically, and `local_data/` is ignored by Git.

Stored fields include a prediction ID, UTC timestamp, model version, submission source, channel, issue category, predicted rating, top probability, combined low-rating probability, and follow-up flag. Raw remarks and detailed raw inputs are not persisted in this database.

To choose another database, set `CSAT_HISTORY_PATH` before launching the application. Relative paths resolve from the server's working directory.

Windows PowerShell:

```powershell
$env:CSAT_HISTORY_PATH = Join-Path (Get-Location) 'local_data/custom_predictions.sqlite3'
.\run_demo.ps1
```

Linux or macOS:

```bash
export CSAT_HISTORY_PATH="$PWD/local_data/custom_predictions.sqlite3"
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

## Model artifacts and preprocessing

Keep these bundled files together in `csat_deployment_artifacts/`:

| File | Purpose |
| --- | --- |
| `champion_csat_ann.keras` | Trained five-class ANN used for inference. |
| `tfidf_vectorizer.joblib` | Fitted text vectorizer. |
| `deployment_metadata.json` | Exact feature order, categories, defaults, caps, class mapping, and model version. |
| `parity_fixture.npz` | Numerical inputs and expected probabilities used by regression checks. |
| `parity_report.json` | Recorded export consistency checks. |

Load only trusted model and joblib artifacts.

The inference pipeline reproduces text cleaning, raw-text counts, complaint keywords, numerical caps, log transforms, cyclical hour features, the handling-to-price ratio, and category encoding in the model's exact **142-feature** order. No additional numerical scaler is used.

Application responsibilities are separated across:

- [app.py](app.py): forms, batch uploads, results, and navigation.
- [deployment/predictor.py](deployment/predictor.py): input validation, feature construction, and inference.
- [deployment/history.py](deployment/history.py): SQLite persistence.
- [deployment/export.py](deployment/export.py): artifact export from trained notebook state.

## Run the integration checks

After installing dependencies, run from the repository root.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux or macOS:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

The tests cover model probability parity, preprocessing, category encoding, invalid inputs, batch predictions, SQLite history, and Streamlit form submissions and reruns. History checks use a temporary database.

The bundled parity report records feature reconstruction checks for **12,887 held-out rows and 142 features**, with a maximum difference of approximately **2.22 × 10⁻¹⁶**. Model probabilities were compared on **128 rows**, with a maximum difference of **0.0**. These checks measure numerical consistency, not predictive accuracy.

## Export an updated model

This step is needed only after training a model for a new deployment bundle.

1. Open the notebook in an environment with its analysis and training dependencies installed. The application requirements file is not a complete notebook environment specification.
2. Run the notebook through champion ANN selection so the trained model, fitted TF-IDF vectorizer, data frames, and train/test features are available in memory.
3. With the repository root importable in that notebook session, run:

```python
from deployment.export import export_from_session

export_from_session(globals())
```

The exporter checks preprocessing and model parity and writes the deployment artifacts. It reads the trained objects from memory and does not retrain the model. Keep the resulting model, vectorizer, metadata, and parity files together as one bundle.

Restart Streamlit after replacing artifacts to clear the cached model, then run the integration checks.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `Project .venv is missing` | Create the Python 3.12 environment in the repository root using the setup commands. |
| PowerShell blocks `run_demo.ps1` | Use the direct Streamlit command above. |
| A dependency cannot be installed | Confirm Python 3.12 is selected and check package availability for the operating system and architecture. |
| Model cannot be loaded | Check that all required artifacts exist and dependencies match the pinned requirements. |
| Model and input schema do not match | Restore a matching deployment bundle or export the model and preprocessing files together. |
| CSV validation fails | Check required column names, supported categories, numeric values, and zero survey lag against the input requirements. |
| Port 8501 is already in use | Stop the existing server or use the direct launch command with `--server.port 8502`, then open http://127.0.0.1:8502. |
| History cannot be saved | Ensure the database directory is writable or set `CSAT_HISTORY_PATH` to a writable location. |

## Evaluation and limitations

The notebook reports **38.73% validation accuracy**, **0.2532 macro F1**, and **44.51% score-1 recall** for the weighted ANN. These are validation metrics, rather than test-set or live application accuracy.

Rare-rating predictions remain weak, class imbalance is substantial, and probabilities are uncalibrated. Some preprocessing was fitted before splitting, and customer feedback may be available only alongside the survey. Price and handling-time caps also remove much of the original variation. The application preserves these preprocessing choices to match the exported model.

The project demonstrates classification of rated interactions. Early-warning use, improvements to preprocessing, and any operational follow-up policy require separate evaluation using actual labels.
