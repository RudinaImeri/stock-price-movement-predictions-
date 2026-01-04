import streamlit as st
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_categorical
from src.model import train_model
from src.evaluate import evaluate_model

st.title("Model Training")


@st.cache_resource
def train_cached_model(X_train, y_train):
    model = train_model(X_train, y_train)
    return model


if st.button("Train Model"):
    with st.spinner("Training model..."):
        train, _ = load_data("data/train.csv", "data/test.csv")
        train = clean_data(train)

        train = train.dropna(subset=["result"])

        X = train.drop(columns=["result", "id"])
        y = train["result"]

        X, encoders = encode_categorical(X)

        st.session_state["X_train"] = X

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.6, random_state=42, stratify=y
        )

        model = train_cached_model(X_train, y_train)

        metrics = evaluate_model(model, X_val, y_val)

        st.success("Training completed ✅")

        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("F1-score", f"{metrics['f1']:.3f}")

        st.session_state["model"] = model
        st.session_state["encoders"] = encoders
        st.session_state["X_train"] = X_train