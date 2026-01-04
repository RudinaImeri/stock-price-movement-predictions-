from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features
from src.model import train_model
from src.evaluate import evaluate_model
import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

train, test = load_data(TRAIN_PATH, TEST_PATH)

train = clean_data(train)
test = clean_data(test)

train = create_features(train)
test = create_features(test)

TARGET = "result"

DROP_COLS = ["result", "id", "exchange"]

X = train.drop(columns=DROP_COLS).select_dtypes(include=["number"])
y = train[TARGET]

X_test = (
    test.drop(columns=["id", "exchange"])
    .select_dtypes(include=["number"])
)
model = train_model(X, y)

accuracy = evaluate_model(model, X, y)
print("Cross-validation accuracy:", accuracy)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "result": predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully")
