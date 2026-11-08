# CSAT Project: Understanding and Video Presentation Guide

**Purpose:** Use this as your personal preparation document and speaking guide. Read the explanations first, rehearse the app steps, and then record the suggested script in your own voice.

**Suggested recording length:** About 10–12 minutes, depending on your speaking speed and time spent clicking. The timings are a guide, not a strict limit.

## 1. The whole idea in simple words

Imagine that a customer contacts an online shopping company's support team about a delayed delivery or a refund. The interaction has information such as the support channel, issue category, response delay and customer remarks.

My project learns patterns from previous interactions that already have satisfaction ratings. It uses these patterns to predict a rating from 1 to 5 for another interaction.

I first explored and prepared the data, then compared two neural networks. I saved the selected ANN and connected it to a Streamlit application. The app lets a user enter one interaction, predict several interactions from a CSV, and review saved prediction history.

The important distinction is that the model is a prototype trained on rated interactions. Whether it can predict dissatisfaction before the customer completes a survey still needs separate testing.

### Your project in one sentence

“I built a customer satisfaction prediction prototype that combines support details and customer remarks, compares two neural networks, and demonstrates the selected model through a local Streamlit app.”

### What you actually completed

| Part | What you did |
|---|---|
| Data understanding | Inspected 85,907 interactions and 20 original columns. |
| Data checking | Checked missing values and duplicates; found strong rating imbalance. |
| Exploration | Examined ratings, support groups, timing and available price/handling measurements. |
| Statistical testing | Investigated relationships involving price, agent tenure and support channel. |
| Preparation | Filled missing values, cleaned remarks, encoded categories and created timing/text features. |
| Modeling | Trained a baseline MLP and a deeper ANN with class weights. |
| Comparison | Checked accuracy, macro F1 and rating-1 recall. |
| Interpretation | Examined useful features by shuffling them and measuring accuracy changes. |
| Export | Saved the trained ANN, text vectorizer and feature metadata. |
| Application | Added single prediction, batch prediction and local history. |

## 2. Words you should understand before recording

| Term | Simple meaning in this project |
|---|---|
| CSAT | Customer Satisfaction; the rating being predicted, from 1 to 5. |
| Supervised learning | Learning from examples that already have the correct answer. |
| Classification | Choosing a category; here, one of five ratings. |
| Feature | An input used by the model, such as response delay or a text-derived number. |
| Target / label | The actual CSAT rating used as the answer during training and evaluation. |
| ANN | Artificial Neural Network; layers of calculations that learn patterns from examples. |
| MLP | Multilayer Perceptron; also a neural network, used here as the simpler baseline. |
| Training | Adjusting the model using historical examples and their actual labels. |
| Prediction / inference | Using the already trained model to estimate a rating for an input. |
| Class imbalance | Some ratings occur far more often than others. |
| Class weights | Giving mistakes on rare ratings more importance during training. |
| TF-IDF | A method that turns words and word pairs into numerical text features. |
| One-hot encoding | Turning a category into 0/1 indicators that the model can process. |
| Epoch | One pass through the training data. |
| Early stopping | Stopping training when validation loss stops improving. |
| Dropout | Temporarily ignoring some calculation units during training to reduce over-reliance. |
| Validation data | Examples used to compare models and guide training. |
| Test data | Separate examples reserved for final evaluation after model selection. |
| Streamlit | The Python framework used to build the app interface. |
| SQLite | The local database used to save prediction summaries. |

You do not need to explain every term in the video. Use the definitions if a teacher asks a follow-up question.

## 3. How information moves through the project

### During training

Historical support records plus actual ratings → data preparation → 142 numerical inputs → model training → validation comparison → selected model saved to files.

The model sees the actual ratings during training so it can learn from its mistakes.

### During app use

Form entry or CSV → input validation → the same feature preparation → saved ANN → five probabilities → highest-probability rating → follow-up rule → saved prediction summary.

The actual CSAT rating is not an input to the prediction. Extra columns, including an optional actual rating in a CSV, are ignored by the predictor.

### What the main files do

| File or folder | Its job |
|---|---|
| E_Commerce_Customer_Satisfaction_Score_Prediction.ipynb | Contains the data analysis, preparation, training and recorded evaluation. |
| app.py | Builds the app's form, tabs, output charts and buttons. |
| deployment/predictor.py | Checks inputs, builds model features and calls the ANN. |
| deployment/history.py | Saves and reads prediction summaries in SQLite. |
| deployment/export.py | Exports trained objects from the notebook session and checks consistency. |
| csat_deployment_artifacts/champion_csat_ann.keras | Contains the saved neural network. |
| csat_deployment_artifacts/tfidf_vectorizer.joblib | Contains the fitted method for converting remarks into text numbers. |
| csat_deployment_artifacts/deployment_metadata.json | Stores expected feature order, categories, defaults and model details. |
| csat_deployment_artifacts/parity_report.json | Records checks that exported calculations match the notebook session. |
| demo_interactions.csv | Provides synthetic example inputs and the batch format. |
| local_data/predictions.sqlite3 | Stores local prediction history after submissions. |
| run_demo.ps1 | Starts the local Streamlit server. |

Think of the saved model as the learned decision system, the vectorizer as the text converter, and the metadata as the instructions for arranging its inputs correctly.

## 4. Start the app before recording

1. Open the project in PyCharm.
2. Open its PowerShell terminal.
3. Check that the terminal is in D:\workspace\ecommerce-customer-csat-score-prediction.
4. Run: .\run_demo.ps1
5. Wait for the Streamlit server to start.
6. Open http://127.0.0.1:8501 in your browser.
7. Leave the terminal running while using the app. Stop it with Ctrl+C after the demonstration.

If PowerShell blocks the script, use the direct command documented by the project:

.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

Use these as manual rehearsal steps. This guide does not indicate that the server has been started for you.

### Rehearsal checklist

- Open the notebook near the rating distribution and model comparison outputs.
- Open the app and try all three tabs once.
- Confirm that a single prediction and a batch prediction complete successfully.
- Keep demo_interactions.csv ready if you want to demonstrate upload.
- Increase browser zoom if the recording text looks small.
- Close unrelated tabs and hide notifications.
- Remember that rehearsal predictions remain in history. The dashboard may already have records.
- Do not retrain the notebook during the recording; show its saved outputs.
- Keep the results note in Section 9 nearby so you use one consistent set of numbers.

## 5. What every app input means

| App field | Meaning / what to say |
|---|---|
| Load a synthetic example | Fills the form with a prepared demonstration scenario. These examples have no evaluation labels. |
| Support channel | How support communicated with the customer: Inbound, Outcall or Email. |
| Issue category | The main reason for contact, such as a return or refund. |
| Product category | The product group connected with the interaction. |
| Agent tenure | The support agent's experience group; “On Job Training” is one available group. |
| Agent shift | The shift in which the interaction took place. |
| Response delay (hours) | Time from issue reporting to the support response. |
| Order-to-issue delay (days) | Time from placing the order to reporting the issue. |
| Item price (dataset units) | The recorded item price. The model preparation caps it at 979. |
| Handling time (dataset units) | The recorded handling measurement. Its source unit is not established; preparation caps it at 427. |
| Issue reported hour | Hour of day from 0 to 23. |
| Reported on a weekend | Whether the issue was reported on Saturday or Sunday. |
| Customer remarks | The text converted into TF-IDF, word/character counts and keyword features. |

Survey lag is supplied internally as zero because it was constant zero in the exported training features. It is not a visible form control.

These controls provide the raw information. The app converts them into 142 model features: 100 text features, 30 category indicators and 12 numerical or engineered features.

## 6. Understand the app outputs

| Output | What it means |
|---|---|
| Predicted CSAT | The rating with the highest model probability, displayed out of 5. |
| Top model probability | The largest of the five output probabilities. This is not the model's measured accuracy. |
| Probability chart | Shows how the model distributes probability across ratings 1–5. |
| Follow-up suggested | Appears when the predicted rating is 1 or 2. It is an app rule. |
| Last submitted prediction | The output belongs to the latest submitted form, even if you have since edited an input. |
| Interactions scored | Number of saved prediction records, including repeated submissions. |
| Mean predicted rating | Average of saved predicted ratings; it is not average actual customer satisfaction. |
| Flagged for follow-up | Share of saved records whose predicted rating is 1 or 2. |

Example only: if five probabilities were 0.10, 0.15, 0.20, 0.30 and 0.25, the predicted rating would be 4 because 0.30 is largest. Do not read this as an actual result from the app.

The probabilities have not been calibrated. A displayed 60% probability does not establish that the prediction is correct 60% of the time.

## 7. Ready-to-speak video script

Read the quoted paragraphs naturally. The “On screen” instructions are actions for you; they are not part of the narration.

### Part 1 — Introduction, about 45 seconds

**On screen:** Show the project title in the notebook or report.

“Hello, my name is [your name]. My project is E-Commerce Customer Satisfaction Score Prediction using an Artificial Neural Network.

The idea is to study customer-support interactions and estimate the satisfaction rating a customer may give, from 1 to 5. A low predicted score could help a support team identify interactions that deserve closer review.

In this project, I explored the data, prepared numerical and text features, compared two neural networks, and built a local application to demonstrate the selected model.”

### Part 2 — Dataset and main challenge, about 60 seconds

**On screen:** Show dataset shape, missing-value summary and rating distribution.

“The dataset contains 85,907 rated support interactions and 20 original columns. It includes support channel, issue category, customer remarks, timing information, product details and the actual CSAT score.

The main challenge is that the ratings are very uneven. About 69.4% are rating 5, while ratings 2 and 3 are much less common. So a model could get fairly high accuracy simply by predicting 5 most of the time.

There are also many missing values. Only 242 interactions have a recorded handling time. For exploration, I kept recorded measurements separate from the values filled in for modeling.”

### Part 3 — Exploration and preparation, about 75 seconds

**On screen:** Show selected analysis charts, then text-feature and splitting sections.

“I examined how ratings varied across support details and available numerical measurements. I also tested relationships involving price, agent experience and support channel. These showed associations, but they do not prove cause and effect.

For modeling, I filled missing values, cleaned the customer remarks, and converted categories into zero-and-one columns. I used TF-IDF to turn words and word pairs into 100 numerical text features.

I also created response delay, order-to-issue delay, time-of-day and weekend features. Altogether, the models received 142 numerical inputs.

The data was divided into approximately 70% training, 15% validation and 15% test data, while maintaining similar rating proportions.”

### Part 4 — Models and results, about 90 seconds

**On screen:** Show the saved executed comparison table, or the consistent table in PROJECT_REPORT.md.

“I compared two neural networks. The baseline had hidden layers of 64 and 32 units. The deeper ANN had layers of 256, 128 and 64 units, with class weights to give more importance to rare ratings.

The baseline achieved 71.76% validation accuracy. However, it had zero recall for ratings 2, 3 and 4. This means it did not correctly identify any actual examples of those ratings.

The weighted ANN achieved 38.73% accuracy, a macro F1 score of 0.2532, and 44.51% recall for rating 1. Macro F1 gives each rating equal importance. Rating-1 recall tells us how many very dissatisfied customers were correctly identified.

I selected the ANN because it performed slightly better across ratings and identified more rating-1 interactions. This involved a clear tradeoff: overall accuracy was lower, and rare-rating performance was still weak.”

### Part 5 — Introduce the running app, about 30 seconds

**On screen:** Switch to http://127.0.0.1:8501. Show the page title and three tabs.

“This is the Streamlit demonstration app. It loads the saved ANN and preprocessing files, so it does not train a new model whenever I click a button.

There are three sections: Single prediction, Batch predictions and History. I will demonstrate each one.”

### Part 6 — Single prediction, about 90 seconds

**On screen:** Select Single prediction, choose Positive support, point to the input groups and click Predict CSAT.

“I can load a synthetic example or adjust the support details manually. These fields describe the channel, issue, product group, agent experience and shift. The timing fields describe response delay and when the issue occurred. The customer remarks provide text information.

When I click Predict CSAT, the application checks the inputs, prepares the same features expected by the saved model, and calculates probabilities for the five ratings.

The predicted score is the rating with the highest probability. The chart shows all five probabilities. The top probability is the model's output for this example; it is different from validation accuracy.”

**On screen:** Read the actual displayed score. Select Delayed refund, click Predict CSAT and compare. Optionally edit one input and submit again.

“For this submitted example, the predicted rating is [read the screen], and the top model probability is [read the screen].

I will now load a different example. These are synthetic inputs for demonstration, so there is no actual survey score here to check correctness.

Changing the remarks or delay may change the output. The model does not guarantee that every delay increase lowers the score. I need to submit the form again after editing to get an updated prediction.”

**If a follow-up warning appears, say:** “The follow-up message appears because the predicted rating is 1 or 2. This is a simple demonstration rule.”

**If no warning appears, say:** “No follow-up warning appears for this prediction. That warning is triggered only when the predicted rating is 1 or 2.”

### Part 7 — Batch predictions, about 60 seconds

**On screen:** Open Batch predictions. Keep Use the bundled synthetic demo batch selected. Click Run batch predictions. Show the table and Download predictions.

“The batch section handles several interactions together. For this demonstration, I am using the bundled synthetic examples.

The app prepares each row, runs the saved model and displays the predicted rating, probabilities and follow-up flag. I can download the results as a CSV file.

For another dataset, I would download the template, keep its required column names, upload my CSV, uncheck the bundled-demo option, and run the predictions. Uploading a file does not train the model.”

If time is short, demonstrate the bundled batch only. You can explain upload without performing both workflows.

### Part 8 — History and model details, about 60 seconds

**On screen:** Open History, click Refresh dashboard if needed, then show the metrics, chart and table. Open About the model.

“The History tab shows saved prediction summaries. It displays how many records have been scored, the mean predicted rating and the proportion flagged for follow-up.

The chart shows the distribution of predicted ratings. These are model predictions, so this chart does not measure live accuracy or actual customer satisfaction.

The history is saved locally in SQLite and remains after restarting the app. Repeated submissions create additional records. Raw customer remarks are not stored in this history.

The About the model section explains the 142 inputs and important restrictions, including the numerical caps and uncalibrated probabilities.”

### Part 9 — Limitations and closing, about 60 seconds

**On screen:** Return to the report's limitations or keep About the model open.

“This project demonstrates the full process from data analysis to a working prediction interface. The main learning is that high accuracy can hide poor performance on less common ratings.

The current model still has limitations. Many measurements are missing, rare ratings are difficult to predict, and some preprocessing was fitted before the data split. Some customer remarks may also be available only when the survey is collected.

Before practical use, I would improve the preprocessing, evaluate the full held-out test set, check feature availability and collect more representative labeled data.

Thank you. This completes my project explanation and application demonstration.”

## 8. How to explain the app's internal work without complex code

Say this if asked “What happens after clicking Predict?”

“First, the app checks that the values and categories are valid. It cleans the remarks and converts them using the saved text vectorizer. It also calculates timing and other numerical features and converts categories into indicator columns.

It arranges all 142 values in the exact order expected by the ANN. The model produces five probabilities, and the largest one determines the rating. The app then applies the low-score follow-up rule, displays the result, and saves a prediction summary.”

The model is loaded once through Streamlit's resource cache. Normal interface reruns do not reload and retrain it from scratch. The application also reuses the fitted text vocabulary rather than learning a new vocabulary for every upload.

## 9. Keep the results consistent during your recording

Use this result set from the saved executed notebook outputs, matching PROJECT_REPORT.md:

| Measure | Baseline MLP | Weighted ANN |
|---|---:|---:|
| Validation accuracy | 71.76% | 38.73% |
| Macro F1 | 0.2405 | 0.2532 |
| Rating-1 recall | 25.93% | 44.51% |

The saved executed training log stops at epoch 20 and restores epoch 13.

Some written notebook summaries and DEPLOYMENT.md instead quote ANN accuracy 40.14%, macro F1 0.2588, rating-1 recall 42.73%, and a different epoch count. These conflict with the saved executed outputs.

For preparation: use the executed-output figures consistently, and show the matching output table. If asked about the mismatch, say: “The written summary is inconsistent with the saved executed output. My report uses the executed output, and a clean rerun is needed to reconcile the project files.”

Do not describe these figures as a newly measured score of the running app. The exported app's validation accuracy was not independently remeasured for the report.

## 10. Likely questions and simple answers

### Why use deep learning?

“The project explores whether a neural network can combine text features and structured support information. I compared a simple network and a deeper weighted network. These experiments do not prove deep learning is better than every other model; simpler classifiers should also be compared in future work.”

### Why select a model with lower accuracy?

“The ratings are imbalanced. The baseline mainly recognized the common rating and missed several other ratings. The weighted ANN had slightly higher macro F1 and much better rating-1 recall, although its accuracy was lower.”

### Does the model understand customer emotions like a person?

“No. It uses numerical patterns in words, timing and support details. TF-IDF represents words and word pairs, but it does not provide human-level understanding of language.”

### Why does a negative example sometimes get a high rating?

“The model has limited performance and learns several features together. A negative word is only one input. These are synthetic scenarios, and a reasonable-looking example does not guarantee a particular output.”

### Does changing a field immediately update the result?

“No. I must click Predict CSAT again. The result is labeled as the last submitted prediction.”

### Why does changing price above 979 sometimes have no effect?

“Preparation caps price at 979. Values above that limit become the same model input. Handling time is similarly capped at 427.”

### Why are some columns missing from the app?

“The app collects the inputs needed by the exported model. Raw identifiers and staff names were removed, and several raw timestamps were replaced by elapsed durations.”

### Is the highest probability the accuracy?

“No. It is a probability assigned to one example. Accuracy requires comparing many predictions with actual labels.”

### Is the app running on the internet?

“The provided script runs it locally at 127.0.0.1. It is not a publicly hosted service.”

### Does the app learn from history?

“No. History stores summaries for review. It does not automatically update the ANN.”

### Has the test set been fully evaluated?

“The dataset has a separate test partition, but the reviewed notebook does not contain a complete test-set performance report. The model comparison I presented uses validation data.”

### What does the export check prove?

“The recorded check reconstructed 142 features for 12,887 held-out rows and compared reloaded-model probabilities on 128 rows. It supports consistency with the notebook, not prediction accuracy against actual labels.”

### What would you improve first?

“I would fit all preprocessing only on training data, improve missing-value treatment, make the recorded results consistent, compare other models and evaluate the final choice on untouched test data.”

## 11. A shorter 3–4 minute version

If your submission has a short video limit, use this order:

1. State the problem and dataset size.
2. Explain that rating 5 dominates and accuracy alone is insufficient.
3. Mention text conversion, support features and the 142 inputs.
4. Show the model comparison and briefly explain the tradeoff.
5. Make one single prediction and explain the probabilities.
6. Run the bundled batch and show History.
7. End with the missing-data and evaluation limitations.

Short opening: “I built a prototype that predicts a 1-to-5 customer satisfaction rating from support details and customer remarks. I compared two neural networks and connected the selected ANN to a Streamlit app.”

Short results explanation: “The baseline had higher accuracy, but missed ratings 2, 3 and 4. The weighted ANN improved macro F1 and rating-1 recall, with lower overall accuracy.”

Short closing: “The application demonstrates the workflow successfully, but the model needs better data preparation and independent evaluation before real support decisions.”

## 12. Final reminders before pressing Record

- Speak about what the project demonstrated, using the actual screen results.
- Pause after clicking to let viewers read the output.
- Explain one chart at a time; you do not need to describe every code line.
- Use “predicted rating” when discussing the app and history.
- Use “validation results” when discussing the model comparison.
- Do not promise a specific rating for any synthetic scenario.
- Leave the server terminal running throughout the app demonstration.
- Finish by stating what you learned and what needs improvement.

**Project references used:** PROJECT_REPORT.md, app.py, run_demo.ps1, DEPLOYMENT.md, and the recorded model/export evidence already documented in the project report.

