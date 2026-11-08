"""A simple local demo: enter details, predict CSAT, and review history."""
import pandas as pd
import streamlit as st

from deployment.history import read_history, save_predictions
from deployment.predictor import CATEGORICAL, ROOT, Predictor


@st.cache_resource
def load_model():
    # Load once, rather than reloading the ANN after every button click.
    return Predictor()


def predict_and_save(predictor, records, source):
    results = predictor.predict(records)
    save_predictions(records, results, predictor.metadata["model_version"], source)
    return results


def single_prediction(predictor, examples):
    names = ["Positive support", "Delayed refund", "Unresolved delivery"]
    scenario = st.selectbox("Load a synthetic example", names)
    example = examples.iloc[names.index(scenario)]

    with st.form(f"interaction_{scenario}"):
        details, timing = st.columns(2)
        record = {}
        with details:
            labels = ["Support channel", "Issue category", "Product category", "Agent tenure", "Agent shift"]
            for column, label in zip(CATEGORICAL, labels):
                options = predictor.metadata["categories"][column]
                record[column] = st.selectbox(label, options, index=options.index(example[column]))
        with timing:
            record["response_time_hours"] = st.number_input("Response delay (hours)", min_value=0.0, value=float(example.response_time_hours))
            record["order_to_issue_days"] = st.number_input("Order-to-issue delay (days)", min_value=0.0, value=float(example.order_to_issue_days))
            record["item_price"] = st.number_input("Item price (dataset units)", min_value=0.0, value=float(example.item_price))
            record["connected_handling_time"] = st.number_input("Handling time (dataset units)", min_value=0.0, value=float(example.connected_handling_time))
            record["issue_hour"] = st.number_input("Issue reported hour", min_value=0, max_value=23, value=int(example.issue_hour))
            record["is_weekend"] = int(st.checkbox("Reported on a weekend", value=bool(example.is_weekend)))
        record["customer_remarks"] = st.text_area("Customer remarks", value=example.customer_remarks, max_chars=20000)
        record["survey_lag_hours"] = 0.0  # This feature was always zero in training.
        submitted = st.form_submit_button("Predict CSAT", type="primary")

    if submitted:
        try:
            results = predict_and_save(predictor, pd.DataFrame([record]), "single")
            st.session_state["single_result"] = results
        except (ValueError, OSError) as error:
            st.error(str(error))

    if "single_result" in st.session_state:
        result = st.session_state["single_result"].iloc[0]
        st.caption("Last submitted prediction. Submit again after editing the details.")
        score, probability = st.columns(2)
        score.metric("Predicted CSAT", f"{int(result.predicted_csat)} / 5")
        probability.metric("Top model probability", f"{result.top_probability:.1%}")
        if result.follow_up:
            st.warning("Follow-up suggested: predicted CSAT is 1 or 2.")
        probabilities = pd.Series(
            [result[f"probability_csat_{rating}"] for rating in range(1, 6)],
            index=[str(rating) for rating in range(1, 6)], name="Model probability",
        )
        st.bar_chart(probabilities, color="#8970b5")


def batch_predictions(predictor, examples):
    st.write("Upload a CSV using the template, or run the sample batch.")
    st.download_button("Download CSV template", examples.to_csv(index=False), "demo_interactions.csv", "text/csv")
    uploaded = st.file_uploader("Upload interactions CSV", type="csv")
    use_demo = st.checkbox("Use the bundled synthetic demo batch", value=uploaded is None)

    if st.button("Run batch predictions", type="primary"):
        try:
            if use_demo:
                records = examples
            elif uploaded is not None:
                records = pd.read_csv(uploaded)
            else:
                raise ValueError("Upload a CSV or select the demo batch.")
            results = predict_and_save(predictor, records, "batch")
            st.session_state["batch_result"] = pd.concat(
                [records[CATEGORICAL].reset_index(drop=True), results], axis=1,
            )
        except (ValueError, OSError, UnicodeError) as error:
            st.error(str(error))

    if "batch_result" in st.session_state:
        results = st.session_state["batch_result"]
        st.caption("Last successful batch.")
        st.success(f"Scored {len(results)} interactions. {int(results.follow_up.sum())} flagged for follow-up.")
        st.dataframe(results, hide_index=True)
        st.download_button("Download predictions", results.to_csv(index=False), "csat_predictions.csv", "text/csv")


def prediction_history():
    st.button("Refresh dashboard")
    history = read_history()
    if history.empty:
        st.info("Make a prediction to start building history.")
        return

    count, average, follow_up = st.columns(3)
    count.metric("Interactions scored", len(history))
    average.metric("Mean predicted rating", f"{history.predicted_csat.mean():.2f} / 5")
    follow_up.metric("Flagged for follow-up", f"{history.follow_up.mean():.1%}")
    counts = history.predicted_csat.value_counts().reindex(range(1, 6), fill_value=0)
    counts.index = counts.index.astype(str)
    st.bar_chart(counts.rename("Interactions"), color="#8970b5")
    st.dataframe(history.head(100).drop(columns="prediction_id"), hide_index=True)
    st.download_button("Download history", history.to_csv(index=False), "csat_history.csv", "text/csv")
    st.caption("History is saved locally without customer remarks. These are predicted ratings, not actual survey results.")


def select_workspace(page):
    st.session_state["workspace"] = page


def main():
    st.set_page_config(
        page_title="CSAT Predictor", page_icon="💬", layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:
        st.title("💬 CSAT Predictor")
        st.caption("Customer support intelligence")
        st.divider()
        st.caption("WORKSPACE")
        page = st.session_state.setdefault("workspace", "Single prediction")
        for destination in ["Single prediction", "Batch predictions", "History"]:
            st.button(
                destination,
                key=f"nav_{destination}",
                type="primary" if page == destination else "secondary",
                use_container_width=True,
                on_click=select_workspace,
                args=(destination,),
            )
        st.divider()
        st.subheader("About the model")
        st.write("Trained neural network · CSAT ratings 1–5")
        st.caption("Combines customer remarks, support details, and response timing across 142 input features.")
        st.info("Ratings of 1 or 2 are flagged for follow-up.")
        st.caption("Demo predictions support review; they are not actual customer survey ratings.")
    st.title("Customer Satisfaction Predictor")
    st.write("Enter support details and use the trained ANN to predict a CSAT rating from 1 to 5.")
    try:
        predictor = load_model()
    except (OSError, ValueError) as error:
        st.error(f"Could not load the model. See DEPLOYMENT.md. {error}")
        st.stop()

    examples = pd.read_csv(ROOT / "demo_interactions.csv")
    st.subheader(page)
    if page == "Single prediction":
        single_prediction(predictor, examples)
    elif page == "Batch predictions":
        batch_predictions(predictor, examples)
    else:
        prediction_history()
    with st.expander("About the model"):
        st.write("142 inputs: 100 text features, 30 category indicators, and 12 numerical/engineered features.")
        st.write("The original preprocessing caps price at 979 and handling time at 427, and uses zero survey lag. Probabilities are uncalibrated.")
        st.write("This is a demo for rated interactions. Performance before survey collection has not been established. See DEPLOYMENT.md for evaluation limitations.")


if __name__ == "__main__":
    main()
