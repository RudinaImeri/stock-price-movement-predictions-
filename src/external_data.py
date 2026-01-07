import nasdaqdatalink as ndl
import streamlit as st


# ✅ set API key
ndl.ApiConfig.api_key = st.secrets["NASDAQ_API_KEY"]


def load_macro_data():
    gdp = ndl.get("FRED/GDP")
    inflation = ndl.get("FRED/CPIAUCSL")
    return gdp, inflation