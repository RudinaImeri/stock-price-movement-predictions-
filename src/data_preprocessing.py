import pandas as pd


def prepare_api_data(
    df,
    horizon=2,
    buy_th=0.015,
    sell_th=-0.015,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "prepare_api_data expects a pandas DataFrame"
        )

    df = df.copy()
    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    numeric_cols = [
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "volume",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}"
            )
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["ret_1"] = df["price_close"].pct_change()
    df["ret_2"] = df["price_close"].pct_change(2)
    df["ret_3"] = df["price_close"].pct_change(3)

    df["volatility_3"] = (
        df["ret_1"]
        .rolling(3)
        .std()
    )

    df["sma_5"] = (
        df["price_close"]
        .rolling(5)
        .mean()
    )
    df["sma_10"] = (
        df["price_close"]
        .rolling(10)
        .mean()
    )

    df["sma_ratio"] = (
        df["sma_5"] / df["sma_10"]
    )

    df["momentum_5"] = (
        df["price_close"]
        - df["price_close"].shift(5)
    )

    df["future_return"] = (
        df["price_close"]
        .shift(-horizon)
        / df["price_close"]
        - 1
    )

    df["result"] = 1
    df.loc[
        df["future_return"] >= buy_th,
        "result",
    ] = 2
    df.loc[
        df["future_return"] <= sell_th,
        "result",
    ] = 0

    df = df.dropna().reset_index(drop=True)

    return df
