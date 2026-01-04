import streamlit as st
import pandas as pd
import numpy as np

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_categorical

st.title("Predictions")

if "model" not in st.session_state or "encoders" not in st.session_state:
    st.warning("Please train the model first in the **Model Training** page.")
    st.stop()

model = st.session_state["model"]
encoders = st.session_state["encoders"]

_, test = load_data("data/train.csv", "data/test.csv")
test = clean_data(test)

if "id" not in test.columns:
    st.error("Test data must contain an 'id' column")
    st.stop()

X_test = test.drop(columns=["id"], errors="ignore")

X_test, _ = encode_categorical(X_test, encoders)

if st.button("Run Predictions"):
    with st.spinner("Predicting... please wait"):

        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            confidence = np.max(probs, axis=1)
        else:
            confidence = np.ones(len(preds))

    class_map = {
        0: "Sell",
        1: "Hold",
        2: "Buy"
    }

    submission = pd.DataFrame({
        "id": test["id"].values,
        "prediction": [class_map[int(p)] for p in preds],
        "confidence": confidence.round(3)
    })

    submission["action"] = submission["prediction"].apply(
        lambda x: "Sell" if x == "Sell" else "Buy / Hold"
    )

    st.success(f"Predictions completed for {len(submission)} samples")

    rows = st.slider(
        "Preview rows",
        min_value=5,
        max_value=min(500, len(submission)),
        value=20
    )

    st.dataframe(submission.head(rows), width='stretch')

    st.download_button(
        "Download submission.csv",
        data=submission.to_csv(index=False),
        file_name="submission.csv",
        mime="text/csv"
    )
