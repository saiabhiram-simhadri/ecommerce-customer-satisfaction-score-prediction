# E-Commerce Customer Satisfaction Score Prediction

Predict customer satisfaction (CSAT) ratings from **1 to 5** using customer-support details, interaction timing, and customer remarks. This project covers exploratory data analysis, statistical testing, feature engineering, neural-network comparison, and a local Streamlit demonstration.

[Explore the notebook](E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb) · [Project report (Word)](docs/csat_prediction_workflow_report.docx) · [Deployment guide](DEPLOYMENT.md)

## Project overview

The project studies **85,907 rated support interactions** in the Shopzilla e-commerce support context. It compares a baseline multilayer perceptron (MLP) with a deeper Artificial Neural Network (ANN) trained using class weights.

The aim is to identify interactions that may need closer review or follow-up. The experiment evaluates classification of already rated interactions, performance before survey collection requires separate evaluation.

## Dataset and analysis

The source dataset contains **20 original columns** covering support channels, issue categories, product details, agent tenure and shift, timestamps, customer remarks, and the target `CSAT Score`.

- **Channels:** Inbound, Outcall, and Email.
- **Rating distribution:** approximately 69.4% score 5, 13.1% score 1, 13.1% score 4, 3.0% score 3, and 1.5% score 2.
- **Available measurements:** 28,742 customer remarks, 17,206 item prices, and 242 handling times, only 104 interactions have both price and handling time.
- **Analysis:** missing-value assessment, rating distributions, support-group comparisons, timing patterns, and statistical hypothesis tests.

The notebook loads its source CSV from a shared Google Drive URL. The raw training dataset is not bundled in this repository. The included [demo CSV](demo_interactions.csv) contains synthetic examples for the application.

## Modeling workflow

1. Inspect data quality and preserve recorded measurements separately for exploratory analysis.
2. Fill missing values and prepare text, categories, and timing features.
3. Build **142 numerical inputs**, including **100 TF-IDF features** from words and word pairs.
4. Split the data with stratification and random seed **47**.
5. Train and compare the baseline MLP and weighted ANN.
6. Interpret the ANN with permutation importance and export its deployment artifacts.

| Partition | Interactions | Approximate share |
| --- | ---: | ---: |
| Training | 60,131 | 70% |
| Validation | 12,889 | 15% |
| Test | 12,887 | 15% |

The baseline uses hidden layers of **64 and 32 neurons**. The weighted ANN uses **256, 128, and 64 neurons**, Batch Normalization, Dropout, and a five-class output. Training uses a batch size of **128** and a maximum of **40 epochs**. Early stopping ended training after **20 epochs** and restored the weights from **epoch 13**.

Features include cleaned remarks, text lengths, complaint keywords, TF-IDF values, category indicators, log-transformed numerical values, cyclical hour features, and elapsed durations. No additional numerical scaler is applied to the ANN inputs.

## Validation results

| Metric | Baseline MLP | Weighted ANN |
| --- | ---: | ---: |
| Accuracy | 71.76% | 38.73% |
| Macro F1 | 0.2405 | 0.2532 |
| CSAT-1 recall | 25.93% | 44.51% |

The **weighted ANN** was selected for its higher macro F1 and better recognition of score-1 interactions. The baseline has higher overall accuracy but zero recall for ratings 2, 3, and 4. The ANN improves recognition across ratings at the cost of lower exact-rating accuracy, rare-rating predictions remain weak.

These are **validation results**, not test-set or live application accuracy. Macro F1 gives each rating equal importance, while CSAT-1 recall measures the share of actual score-1 interactions correctly identified.

## Local application

The Streamlit application loads the bundled model and preprocessing artifacts without retraining.

- **Single prediction:** enter one interaction and view its predicted CSAT rating and five class probabilities.
- **Batch predictions:** upload a CSV or use the synthetic demo batch, then download predictions.
- **History:** review persisted prediction summaries and the distribution of predicted ratings.
- **Follow-up flag:** predicted ratings 1–2 receive a demonstration follow-up flag.

### Requirements

- Python **3.12** for the documented deployment environment.
- Git to clone the repository.
- Dependencies pinned in [requirements-deployment.txt](requirements-deployment.txt).

### Setup and launch (Windows PowerShell)

```powershell
git clone https://github.com/saiabhiram-simhadri/ecommerce-customer-satisfaction-score-prediction.git
cd ecommerce-customer-satisfaction-score-prediction
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deployment.txt
.\run_demo.ps1
```

Open **http://127.0.0.1:8501**. Stop the application with **Ctrl+C**. After dependencies are installed, normal application use works offline with the bundled artifacts.

If PowerShell blocks the launch script, run Streamlit directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

### Batch input and saved history

Use [demo_interactions.csv](demo_interactions.csv) as the input template. Supply the category values shown in the application and elapsed durations rather than raw dataset timestamps. `survey_lag_hours` must be **0**. Batch uploads support up to **10 MB** and **10,000 interactions**.

Prediction summaries persist in `local_data/predictions.sqlite3`, which is ignored by Git. Raw remarks and detailed raw inputs are not stored in the history database. Set `CSAT_HISTORY_PATH` to use another database location.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full input schema, defaults, supported categories, export workflow, and troubleshooting.

## Repository structure

```text
.
├── E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb  # Analysis and modeling
├── app.py                                               # Streamlit interface
├── deployment/
│   ├── predictor.py                                     # Validation and inference
│   ├── history.py                                       # SQLite prediction history
│   └── export.py                                        # Export from notebook state
├── csat_deployment_artifacts/                            # Model, vectorizer, metadata, parity fixture
├── docs/
│   └── csat_prediction_workflow_report.docx              # Submission report
├── tests/test_deployment.py                              # Integration checks
├── demo_interactions.csv                                # Synthetic batch examples
├── requirements-deployment.txt                           # Pinned application dependencies
├── run_demo.ps1                                          # Local launch script
└── DEPLOYMENT.md                                         # Detailed application guide
```

## Notebook exploration

Open the [notebook](E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb) to review the analysis, model architecture, evaluation outputs, and charts. GitHub can display its saved outputs without running the training workflow.

To execute the analysis, use a Jupyter environment with the notebook's imported packages, including SciPy, Matplotlib, and Seaborn alongside the modeling dependencies. The deployment requirements file supports the application and is not a complete notebook environment specification. Loading the source dataset requires network access.

## Verification

Run the integration checks from the repository root after installing the deployment dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks cover preprocessing, input validation, category encoding, batch predictions, model probability parity, SQLite history, and Streamlit form interactions.

The bundled [parity report](csat_deployment_artifacts/parity_report.json) records feature reconstruction checks across **12,887 held-out rows and 142 features**, with a maximum difference of approximately **2.22 × 10⁻¹⁶**. Reloaded model probabilities were compared on **128 rows**, with a maximum difference of **0.0**. These checks measure numerical consistency, rather than prediction accuracy against actual ratings.

## Limitations and future work

- Strong class imbalance and weak performance on rare ratings limit reliable five-class predictions.
- Extensive missing price and handling-time data, followed by imputation and capping, removes substantial numerical variation.
- Some preprocessing, including TF-IDF vocabulary fitting, occurs before splitting, future evaluation should fit preprocessing on training data only.
- Remarks or survey-related information may be unavailable before the rating is collected, so early-warning use needs separate validation.
- Model probabilities are uncalibrated, and the follow-up flag is a demonstration rule.

Future work includes improving missing-value treatment, evaluating the selected model on the test partition, comparing simpler and ordinal classifiers, calibrating probabilities, and measuring performance using actual labels in a live trial.

## Project documents

- [Analysis and modeling notebook](E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb)
- [Project report (Word)](docs/csat_prediction_workflow_report.docx)
- [Deployment and demonstration guide](DEPLOYMENT.md)
- [Repository on GitHub](https://github.com/saiabhiram-simhadri/ecommerce-customer-satisfaction-score-prediction)
