import streamlit as st
import pandas as pd
from src.api_data import load_market_data_from_api
from src.data_preprocessing import prepare_api_data
from src.ui_theme import apply_global_theme, app_style

apply_global_theme()
st.title("Market Predictions")

STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "ONDS": "Ondas Holding Inc."
}

model = st.session_state.get("model")
features = st.session_state.get("features")
trained_stocks = st.session_state.get("trained_stocks")

if model is None:
    st.warning("Train model first")
    st.stop()

stock_map = {s: i for i, s in enumerate(trained_stocks)}

rows = []
class_map = {0: "Sell", 1: "Hold", 2: "Buy"}

for symbol, name in STOCKS.items():

    df = prepare_api_data(load_market_data_from_api(symbol))

    df["stock"] = symbol
    df["stock_id"] = stock_map[symbol]

    # numeric safety
    numeric_cols = [
        "price_open",
        "price_close",
        "price_high",
        "price_low",
        "volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # EXACT same features
    X = df[features]

    # USE ONLY LAST DAY
    X_last = X.iloc[[-1]]
    last_row = df.iloc[-1]

    pred = int(model.predict(X_last)[0])

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X_last).max()
    else:
        confidence = 1.0

    rows.append({
        "Symbol": symbol,
        "Name": name,
        "Price": round(float(last_row["price_close"]), 2),
        "Change": round(
            float(last_row["price_close"] - last_row["price_open"]), 2
        ),
        "Change %": round(
            float(
                (last_row["price_close"] - last_row["price_open"])
                / last_row["price_open"] * 100
            ), 2
        ),
        "Volume": int(last_row["volume"]),
        "Prediction": class_map[pred],
        "Confidence": round(confidence, 3),
    })

table = pd.DataFrame(rows)

st.dataframe(
    app_style(table),
    use_container_width=True
)
