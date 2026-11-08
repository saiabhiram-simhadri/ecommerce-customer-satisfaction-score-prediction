# E-Commerce Customer Satisfaction Score Prediction
## Project Report

**Project type:** Supervised Machine Learning and Deep Learning  
**Project context:** Shopzilla E-Commerce Support Operations  
**Submitted by:** [Your name]  
**Roll number / Student ID:** [Your ID]  
**Institution / Department:** [Your institution and department]  
**Submitted to:** [Faculty / Guide name]  
**Submission date:** [Date]

## Abstract

This project studies how customer-support records can be used to predict Customer Satisfaction (CSAT) ratings from 1 to 5. It uses 85,907 rated interactions containing support details, timing information and customer remarks. The work covers data checking, exploratory analysis, statistical testing, feature preparation, neural-network training, model comparison and a local demonstration application.

Two neural networks were compared. The baseline model achieved 71.76% validation accuracy, but failed to identify any actual examples of ratings 2, 3 and 4 correctly. The weighted Artificial Neural Network (ANN) achieved 38.73% accuracy, a macro F1 score of 0.2532 and 44.51% recall for rating 1 in the notebook's saved executed evaluation. It was selected because it gave more attention to the less common ratings and identified more very dissatisfied customers. Its overall accuracy was lower, and its predictions remain limited.

The result is a working research prototype. Further evaluation is needed before using it to make real customer-support decisions.

## 1. Introduction and Problem Statement

Customer satisfaction helps an e-commerce business understand the quality of its support service. A customer may contact support about a refund, return, cancellation, payment or product query. After the interaction, the customer can provide a CSAT rating.

In this project, CSAT is a five-level scale: 1 is the lowest rating and 5 is the highest. Predicting a low rating could help a support team decide which interactions need closer review or follow-up.

The project investigates whether support details and customer remarks contain enough information to predict these ratings. The business aim is earlier identification of dissatisfaction. However, the available dataset contains interactions that already have ratings, and some remarks may have been collected with the survey. The present experiment therefore demonstrates classification of rated interactions; it does not establish an effective early-warning system.

## 2. Objectives

- Understand the structure and quality of the support dataset.
- Explore differences in satisfaction across recorded customer and support details.
- Convert text, categories and time information into numerical model inputs.
- Compare a basic neural network with a deeper ANN using class weights.
- Evaluate both overall accuracy and recognition of less common ratings.
- Save the selected model and demonstrate predictions through a simple application.

## 3. Dataset Description

The notebook loads a shared CSV dataset containing 85,907 rows and 20 original columns. Each row represents one rated support interaction. The channels are Inbound, Outcall and Email.

| Information group | Examples |
|---|---|
| Interaction details | Support channel, issue category and sub-category |
| Customer feedback | Customer remarks and CSAT score |
| Timing | Order time, issue reporting time, response time and survey date |
| Product details | Product category and item price |
| Support staff details | Agent, supervisor, manager, tenure group and shift |
| Identifiers | Interaction ID and order ID |

The target variable is CSAT Score. Identifiers and staff names were removed from the modeling data rather than treated as meaningful numerical predictors.

### 3.1 Data quality

No fully duplicated rows were found. Missing information was a major issue:

| Field | Recorded values | Missing share |
|---|---:|---:|
| Customer remarks | 28,742 | 66.54% |
| Item price | 17,206 | About 80.0% |
| Connected handling time | 242 | About 99.7% |
| Both price and handling time | 104 interactions | Very limited joint coverage |

A separate copy of the data was used for exploratory analysis. This allowed plots to show recorded measurements instead of values filled in for model training. For example, a price-versus-handling-time chart uses only the 104 interactions with both measurements. It cannot describe all 85,907 interactions.

### 3.2 Rating imbalance

| CSAT rating | Approximate share |
|---|---:|
| 1 | 13.1% |
| 2 | 1.5% |
| 3 | 3.0% |
| 4 | 13.1% |
| 5 | 69.4% |

This uneven distribution is called class imbalance. A model can appear accurate by predicting rating 5 most of the time. Always predicting 5 would be correct for roughly 69.4% of this dataset, while providing no useful detection of lower ratings.

## 4. Tools Used

Python was used for the analysis and application. Pandas and NumPy supported data preparation; Matplotlib and Seaborn were used for charts; SciPy supported statistical tests; and scikit-learn provided text features, data splitting, the baseline model and evaluation measures. TensorFlow/Keras was used for the deeper ANN.

The demonstration uses Streamlit for its interface and SQLite for saved prediction summaries. The trained model and supporting files are stored separately so the application can make predictions without retraining.

## 5. Overall Process

The project followed this sequence:

1. Load the dataset and inspect its columns, missing values and duplicates.
2. Keep recorded measurements separate for exploratory analysis.
3. Study rating distributions, support groups, timing and available numerical data.
4. Test selected relationships suggested by the analysis.
5. Prepare missing values, text, categories and new numerical features.
6. Split the data into training, validation and test partitions.
7. Train and compare two neural networks.
8. Inspect which inputs influence the selected ANN.
9. Save the model and preprocessing files.
10. Build a local application and check that its calculations match the exported model.

## 6. Exploratory Analysis and Statistical Findings

The clearest finding was the large number of rating-5 interactions. This shaped the model evaluation: overall accuracy alone would hide weak performance on lower ratings.

The notebook also examined price, handling time, support channel and agent experience. Because price and handling time are frequently missing, their observed patterns must be interpreted carefully.

Three statistical questions were tested using a 5% significance level. A small p-value indicates evidence of a relationship under the test assumptions. It does not prove that one variable causes another.

| Question and method | Recorded result | Simple interpretation |
|---|---|---|
| Are average recorded prices higher for rating 1 than rating 5? Welch's t-test | Means: 8,848.80 versus 4,713.35; p approximately 4.63 × 10⁻⁴³ | In the recorded-price subset, rating-1 interactions had higher average prices. |
| Does average CSAT differ across agent tenure groups? One-way ANOVA | F = 50.0622; p approximately 3.72 × 10⁻⁴² | Group averages differed, but tenure explained only about 0.23% of score variation. |
| Is support channel related to satisfaction group? Chi-square test | Chi-square = 191.5305; p approximately 2.57 × 10⁻⁴² | Channel and satisfaction were related, but the relationship was weak. |

For the channel test, ratings 1–2 were grouped as dissatisfied and ratings 4–5 as satisfied. Rating 3 was excluded, leaving 83,349 interactions. These findings suggest topics for further investigation rather than proving that changing a channel, product price or agent tenure would improve satisfaction.

## 7. Data Preparation and Feature Engineering

### 7.1 Missing values and unusually large values

Missing remarks and categories were replaced with placeholders, including “No remarks provided” for empty feedback. Missing prices and handling times were filled with medians: 979 and 427 respectively.

The notebook then applied an outlier-capping rule. Because so many values had been replaced by the medians, the resulting upper limits were also 979 for price and 427 for handling time. Many genuine higher measurements were therefore reduced to these limits. This removed considerable variation and is a limitation of the current approach.

The source notebook does not establish the handling-time unit, so this report does not label it as minutes or seconds.

### 7.2 Text preparation

Customer remarks were converted to lowercase, punctuation was removed and repeated spaces were cleaned. The project then created:

- Character and word counts.
- A flag for complaint-related words such as “delay,” “damaged” and “refund.”
- 100 TF-IDF features based on individual words and word pairs.

TF-IDF converts words into numbers based on their frequency within remarks and across the dataset. This lets the neural networks use text alongside support details.

### 7.3 Categories and timing

Channel, issue category, product category, agent tenure and shift were converted into 0/1 indicator columns. For example, an Outcall indicator records whether the interaction belongs to that channel.

Additional features described response delay, time from order to issue report, survey delay, time of day, weekend status and a handling-time/price combination. Time of day was represented in a way that keeps late-night and early-morning hours close together.

Price, handling time and response delay were transformed using log(1 + value), which compresses large values. This reduced response-delay skewness; it did not improve the skewness of every numerical field.

### 7.4 Final model inputs

The final dataset contained 142 numerical input features, including the 100 text features. Raw text, raw timestamps, identifiers and selected detailed category fields were removed after useful information had been extracted.

No additional numerical scaler was applied before model training. A scaler saved separately in the notebook was not part of the ANN's training process and is not used by the demonstration application.

Some preparation steps, including median calculation and TF-IDF vocabulary fitting, took place before splitting. This allows information from the evaluation data to influence preparation and weakens the independence of the reported evaluation.

## 8. Data Splitting and Model Training

The data was split with random seed 47 while keeping similar rating proportions in each partition. This is called a stratified split.

| Partition | Interactions | Purpose |
|---|---:|---|
| Training | 60,131 | Learn model patterns |
| Validation | 12,889 | Compare models and guide training |
| Test | 12,887 | Separate data for final evaluation |

The proportions are approximately 70%, 15% and 15%. The reported model comparison uses the validation partition. A complete test-set performance report is not present in the reviewed notebook outputs.

### 8.1 Baseline neural network

The first model was an MLPClassifier with hidden layers of 64 and 32 neurons. A hidden layer is a group of calculation units that learns combinations of input patterns. This model provided a basic comparison point.

### 8.2 Weighted ANN

The second model used hidden layers of 256, 128 and 64 neurons. Its final layer produced five probabilities, one for each CSAT rating. The rating with the highest probability became the prediction.

Class weights increased the training penalty for mistakes on rare ratings. This encouraged the model to pay more attention to ratings 2 and 3 instead of mostly predicting 5.

Dropout temporarily ignored some calculation units during training to reduce reliance on particular patterns. Batch normalization helped stabilize the calculations. Training also reduced the learning rate when progress slowed and stopped early when validation loss failed to improve.

The model allowed up to 40 epochs, with 128 examples per training batch. In the saved training log, it stopped after epoch 20 and restored weights from epoch 13. An epoch is one pass through the training data.

## 9. Model Results and Selection

### 9.1 Meaning of the measures

**Accuracy** is the share of predictions with the exact correct rating.

**Macro F1** combines precision and recall for each rating, then gives all five ratings equal importance. A higher value indicates better performance on this measure.

**Rating-1 recall** is the share of actual rating-1 interactions correctly identified as rating 1. This helps assess recognition of very dissatisfied customers.

### 9.2 Validation comparison

The following figures come from the notebook's saved executed evaluation and comparison table.

| Measure | Baseline MLP | Weighted ANN |
|---|---:|---:|
| Accuracy | 71.76% | 38.73% |
| Macro F1 | 0.2405 | 0.2532 |
| Rating-1 recall | 25.93% | 44.51% |

The baseline had higher exact-rating accuracy, but zero recall for ratings 2, 3 and 4. It mainly recognized ratings 1 and 5.

The weighted ANN improved macro F1 by 0.0127 and rating-1 recall by 18.58 percentage points. Its accuracy fell by 33.03 percentage points. It was selected because the project valued broader recognition across ratings and better detection of very dissatisfied customers. These results show a tradeoff, rather than an improvement on every measure.

For every 100 genuinely rating-1 interactions, the validation results suggest that the baseline identified about 26 and the weighted ANN about 45. The ANN still missed more than half.

### 9.3 ANN results for individual ratings

| Actual rating | Precision | Recall | F1 score |
|---|---:|---:|---:|
| 1 | 42.64% | 44.51% | 0.4355 |
| 2 | 1.95% | 18.23% | 0.0352 |
| 3 | 3.49% | 4.17% | 0.0380 |
| 4 | 15.93% | 43.73% | 0.2336 |
| 5 | 81.20% | 38.62% | 0.5235 |

Precision means how often a predicted rating was correct. Very low precision for ratings 2 and 3 means many predictions of those ratings were incorrect. The model remains weak for the rare ratings despite class weighting.

### 9.4 Result consistency note

The notebook's written summaries and DEPLOYMENT.md quote a different ANN result: 40.14% accuracy, 0.2588 macro F1 and 42.73% rating-1 recall. The written summary also mentions 22 training epochs and selected epoch 15.

These figures do not match the saved executed evaluation and training log. This report uses the executed outputs consistently: 38.73%, 0.2532, 44.51%, and selected epoch 13 after 20 epochs. These are existing recorded results, not measurements from a new training run. The exported application's validation accuracy was not independently remeasured for this report. A clean, fully documented rerun is needed to bring the notebook narrative, outputs and deployed model results into agreement.

## 10. Understanding the Selected Model

The notebook used permutation importance to study the ANN. One feature was shuffled at a time, and the drop in validation accuracy was measured. A larger drop suggests that the model relies more on that feature.

The calculation used 1,500 validation interactions and repeated each shuffle five times. The saved output ranked remark word count, response delay and the text feature “good” among the leading inputs. Their average accuracy drops were approximately 1.65, 1.57 and 1.00 percentage points respectively.

This suggests that feedback text and response timing help the model distinguish rated interactions. It does not prove that longer remarks or a change in response delay directly causes a particular satisfaction rating.

## 11. Model Saving and Demonstration Application

The selected Keras ANN, fitted TF-IDF vectorizer and feature metadata were saved as separate files. The Streamlit application loads these files and applies the same feature preparation used by the exported model.

The application provides three main functions:

- **Single prediction:** Enter one interaction and view its predicted rating and five probabilities.
- **Batch prediction:** Upload a CSV or use bundled synthetic examples, then download predictions.
- **History:** View saved prediction summaries and the distribution of predicted ratings.

Predicted ratings 1–2 receive a follow-up flag. This is a demonstration rule, not a validated business policy. The displayed probabilities have not been calibrated, so they should not be treated as proven certainty levels.

Prediction summaries are stored locally in SQLite. Raw remarks and detailed raw inputs are not stored in the history database. Normal use does not retrain the model.

The app can be started with the project's run_demo.ps1 script and opened at http://127.0.0.1:8501. Synthetic examples demonstrate the workflow; they are not labeled evidence of accuracy.

### 11.1 Export verification

The saved parity report records checks that the application reproduces the notebook calculations:

| Check | Recorded result |
|---|---|
| Feature reconstruction | 12,887 held-out rows and 142 features checked |
| Largest feature difference | Approximately 2.22 × 10⁻¹⁶, a negligible rounding difference |
| Reloaded-model prediction comparison | 128 rows checked |
| Largest probability difference | 0.0 |

These checks support consistency of feature preparation and model reloading. They do not measure how often predictions match real CSAT ratings.

The repository also contains checks for preprocessing, invalid inputs, categories, batch predictions, history and Streamlit interactions. This report does not claim a new execution of those tests.

## 12. Limitations and Future Improvements

The main limitation is class imbalance: ratings 2 and 3 have too few examples for reliable predictions in the current experiment. Price and handling time also have very limited recorded coverage, and the current filling-and-capping approach removes substantial variation.

Evaluation needs improvement. Preprocessing should be fitted only on training data. The full test partition should then be evaluated once the model is selected. The conflicting saved results should be reconciled through a consistent run.

Feature availability must also be checked. Remarks collected with a survey cannot support predictions made before that survey. Survey delay is constant zero in the exported training features, so the app requires zero for that input.

Future work should collect more representative data, improve missing-value treatment, compare simpler classifiers, evaluate methods that account for the ordered 1–5 rating scale, and check whether predicted probabilities match observed outcomes. Any live trial should measure accuracy and follow-up workload using actual labels collected later.

The current work does not establish a reduction in churn, an increase in satisfaction or improved business revenue.

## 13. Conclusion

This project completed a workflow from exploring customer-support data to demonstrating a saved neural-network model in a local application.

The baseline model achieved higher accuracy but overlooked several ratings. The weighted ANN improved recognition across ratings and detected more rating-1 interactions, at the cost of lower overall accuracy. The saved executed evaluation reports 38.73% accuracy, 0.2532 macro F1 and 44.51% rating-1 recall for the ANN.

The project shows the value of evaluating more than accuracy when ratings are unevenly distributed. It also shows that data quality, feature availability and consistent evaluation are essential before a model can be trusted for practical use.

## 14. Project Sources

This report is based on the local project files and their saved outputs:

1. README.md — project overview and demonstration entry point.
2. E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb — data analysis, feature preparation, model training and saved evaluation outputs.
3. DEPLOYMENT.md — application workflow, input requirements, verification description and limitations.
4. csat_deployment_artifacts/deployment_metadata.json — exported feature schema, defaults, model version and limitations.
5. csat_deployment_artifacts/parity_report.json — recorded export consistency checks.

Project repository: https://github.com/saiabhiram-simhadri/ecommerce-customer-csat-score-prediction

