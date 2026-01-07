import streamlit as st
import pandas as pd
from src.api_data import load_market_data_from_api
from src.data_preprocessing import prepare_api_data
from src.model import train_model
from sklearn.metrics import accuracy_score, f1_score
from src.ui_theme import apply_global_theme

apply_global_theme()
st.title("Model Training (Multi‑Stock)")

STOCK = ["AAPL",
         "MSFT",
         "GOOGL",
         "AMZN",
         "TSLA",
         "NVDA",
         "ONDS"]
FEATURES = [
    "stock_id",
    "ret_1", "ret_2", "ret_3",
    "volatility_3",
    "sma_ratio",
    "momentum_5"
]

if st.button("Train Global Model"):

    all_data = []

    for stock in STOCK:
        raw = load_market_data_from_api(stock)
        df = prepare_api_data(raw)
        df["stock"] = stock
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)
    data["stock_id"] = data["stock"].astype("category").cat.codes

    X = data[FEATURES]
    y = data["result"]

    train_idx = []
    test_idx = []

    for stock in data["stock"].unique():
        df_t = data[data["stock"] == stock]
        split = int(len(df_t) * 0.8)
        train_idx += df_t.index[:split].tolist()
        test_idx += df_t.index[split:].tolist()

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    st.session_state["model"] = model
    st.session_state["features"] = FEATURES

    f1 = f1_score(
        y_test,
        preds,
        average="macro",
    )

    st.metric("Accuracy (test)", f"{accuracy_score(y_test, preds):.3f}")
    st.metric("F1-score (test)", f"{f1:.3f}")
    st.session_state["features"] = FEATURES
    st.session_state["trained_stocks"] = STOCK