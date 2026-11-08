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

The source dataset is included at [data/eCommerce_Customer_support_data.csv](data/eCommerce_Customer_support_data.csv), with **85,907 rows and 20 columns**. Use it for notebook analysis and training. The separate [demo CSV](demo_interactions.csv) contains synthetic examples formatted for application uploads.

## Feature descriptions

The following table describes the original dataset columns using the variable definitions from the notebook.

| Feature | Type | Description |
| --- | --- | --- |
| `Unique id` | String / Alphanumeric | Unique identifier assigned to track each individual customer support ticket or interaction. |
| `channel_name` | String / Categorical | Support channel through which the interaction took place, such as Inbound call, Outcall, or Email. |
| `category` | String / Categorical | High-level functional category of the customer request (e.g., Returns, Order Related, Cancellation, Product Queries). |
| `Sub-category` | String / Categorical | Specific issue or granular service topic under the main category (e.g., Reverse Pickup Enquiry, Installation/demo, Delayed). |
| `Customer Remarks` | String / Text | Free-form qualitative feedback and comments provided by the customer regarding their service experience. |
| `Order_id` | String / Alphanumeric | Unique order identifier associated with the support ticket (missing for some tickets, the reason is not established). |
| `order_date_time` | Datetime / String | Exact date and timestamp when the customer originally placed the purchase order. |
| `Issue_reported at` | Datetime / String | Timestamp when the customer first reported the issue or contacted customer support. |
| `issue_responded` | Datetime / String | Timestamp when a customer support representative responded to or handled the inquiry. |
| `Survey_response_Date` | Datetime / String | Date when the customer submitted their feedback rating and survey response. |
| `Customer_City` | String / Categorical | Registered delivery city or geographic location of the customer. |
| `Product_category` | String / Categorical | Merchandise department or product classification of the purchased item (e.g., Electronics, LifeStyle, Mobile). |
| `Item_price` | Float / Continuous | Purchase monetary value associated with the ticket, currency is not documented in the CSV. |
| `connected_handling_time` | Float / Continuous | Duration taken by the agent to handle the customer interaction, reported in source units, the CSV does not document the duration unit. |
| `Agent_name` | String / Categorical | Name of the customer service representative who handled the interaction. |
| `Supervisor` | String / Categorical | Direct team lead or supervisor overseeing the handling agent. |
| `Manager` | String / Categorical | Support operations manager responsible for the respective support unit. |
| `Tenure Bucket` | String / Categorical | Experience band of the handling agent (e.g., On Job Training, 0-30, 31-60, 61-90, >90 days). |
| `Agent Shift` | String / Categorical | Operational working shift during which the interaction occurred (Morning, Afternoon, Evening, Night, Split). |
| `CSAT Score` | Integer / Ordinal | Customer Satisfaction score given by the customer on a discrete 1-to-5 scale, serving as the primary multi-class target variable. |

The final model input contains **100 TF-IDF features**, **30 category indicators**, and **12 numerical or engineered features**. Category encoding drops the first level of each categorical field. Blank remarks become `No remarks provided`, and blank numerical values use the exported defaults. Supported categories and defaults are documented in [DEPLOYMENT.md](DEPLOYMENT.md).

The target, **`CSAT Score`**, is an integer rating from **1 to 5** and is not supplied as a prediction input. Identifiers, staff names, and raw timestamps are excluded from the final model inputs.

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
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501**. Stop the application with **Ctrl+C**. After dependencies are installed, normal application use works offline with the bundled artifacts.

### Batch input and saved history

Use [demo_interactions.csv](demo_interactions.csv) as the input template. Supply the category values shown in the application and elapsed durations rather than raw dataset timestamps. `survey_lag_hours` must be **0**. Batch uploads support up to **10 MB** and **10,000 interactions**.

Prediction summaries persist in `local_data/predictions.sqlite3`, which is ignored by Git. Raw remarks and detailed raw inputs are not stored in the history database. Set `CSAT_HISTORY_PATH` to use another database location.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full input schema, defaults, supported categories, export workflow, and troubleshooting.

## Repository structure

```text
.
├── E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb  # Analysis and modeling
├── data/
│   └── eCommerce_Customer_support_data.csv               # Source analysis and training dataset
├── app.py                                               # Streamlit interface
├── deployment/
│   ├── predictor.py                                     # Validation and inference
│   ├── history.py                                       # SQLite prediction history
│   └── export.py                                        # Export from notebook state
├── csat_deployment_artifacts/                            # Model, vectorizer, metadata, parity fixture
├── docs/
│   └── csat_prediction_workflow_report.docx              # Submission report
├── demo_interactions.csv                                # Synthetic batch examples
├── requirements-deployment.txt                           # Pinned application dependencies
└── DEPLOYMENT.md                                         # Detailed application guide
```

## Notebook exploration

Open the [notebook](E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb) to review the analysis, model architecture, evaluation outputs, and charts. GitHub can display its saved outputs without running the training workflow.

To execute the analysis, use a Jupyter environment with the notebook's imported packages, including SciPy, Matplotlib, and Seaborn alongside the modeling dependencies. The deployment requirements file supports the application and is not a complete notebook environment specification. The notebook currently loads the dataset from Google Drive. To use the bundled CSV offline, replace its dataset-loading cell with the following code and run the notebook with the repository root as the working directory:

```python
import pandas as pd

cust_support_df = pd.read_csv("data/eCommerce_Customer_support_data.csv")
```

The raw dataset uses the original column names and timestamps. Use `demo_interactions.csv`, rather than the raw training CSV, as the application's batch input template.

## Export verification

The bundled [parity report](csat_deployment_artifacts/parity_report.json) records feature reconstruction checks across **12,887 held-out rows and 142 features**, with a maximum difference of approximately **2.22 × 10⁻¹⁶**. Reloaded model probabilities were compared on **128 rows**, with a maximum difference of **0.0**. These checks measure numerical consistency, rather than prediction accuracy against actual ratings.

## Limitations and future work

- Strong class imbalance and weak performance on rare ratings limit reliable five-class predictions.
- Extensive missing price and handling-time data, followed by imputation and capping, removes substantial numerical variation.
- Some preprocessing, including TF-IDF vocabulary fitting, occurs before splitting, future evaluation should fit preprocessing on training data only.
- Remarks or survey-related information may be unavailable before the rating is collected, so early-warning use needs separate validation.
- Model probabilities are uncalibrated, and the follow-up flag is a demonstration rule.

Future work includes improving missing-value treatment, evaluating the selected model on the test partition, comparing simpler and ordinal classifiers, calibrating probabilities, and measuring performance using actual labels in a live trial.

## Project documents

- [Source dataset](data/eCommerce_Customer_support_data.csv)
- [Analysis and modeling notebook](E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb)
- [Project report (Word)](docs/csat_prediction_workflow_report.docx)
- [Deployment and demonstration guide](DEPLOYMENT.md)
- [Repository on GitHub](https://github.com/saiabhiram-simhadri/ecommerce-customer-satisfaction-score-prediction)
