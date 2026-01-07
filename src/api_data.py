import yfinance as yf
import pandas as pd


def load_market_data_from_api(symbol, period="2y"):
    df = yf.download(symbol, period=period, progress=False)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"No valid data returned for {symbol}")

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing columns for {symbol}")

    df = df.rename(columns={
        "Date": "date",
        "Open": "price_open",
        "High": "price_high",
        "Low": "price_low",
        "Close": "price_close",
        "Volume": "volume",
    })

    return df[[
        "date",
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "volume"
    ]]
