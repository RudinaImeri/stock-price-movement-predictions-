import streamlit as st
import pandas as pd
from src.api_data import load_market_data_from_api
from src.data_preprocessing import prepare_api_data
from src.ui_theme import apply_global_theme

apply_global_theme()

st.title("Training Data Overview")

STOCK = ["AAPL",
         "MSFT",
         "GOOGL",
         "AMZN",
         "TSLA",
         "NVDA",
         "ONDS",
         ]

all_data = []

for stock in STOCK:
    raw = load_market_data_from_api(stock)
    df = prepare_api_data(raw)
    df["stock"] = stock
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

st.subheader("Preview: 5 rows per stock")

preview = (
    data
    .groupby("stock", group_keys=False)
    .apply(lambda x: x.head(5))
)

# Put stock as first column
cols = ["stock"] + [c for c in preview.columns if c != "stock"]
preview = preview[cols]

st.dataframe(preview, use_container_width=True)

st.subheader("Rows per stock (training data balance)")
st.dataframe(
    data["stock"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "stock", "stock": "Rows"})
)

st.write("Total rows:", len(data))
st.write("Number of stocks:", data["stock"].nunique())
