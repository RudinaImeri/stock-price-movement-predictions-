import pandas as pd


def load_data(train_path, test_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def clean_data(df):
    df = df.copy()
    df.ffill(inplace=True)
    return df
