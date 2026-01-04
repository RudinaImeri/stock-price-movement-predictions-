from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "RSI_signal",
    "Stoch_O_signal",
    "ichimoku_c_signal",
    "fibonacci_signal",
    "Bollinger_signal",
    "exchange",
]


def encode_categorical(df, encoders=None):
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if col not in encoders:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                le = encoders[col]
                df[col] = df[col].astype(str)
                df[col] = le.transform(df[col])

    return df, encoders
