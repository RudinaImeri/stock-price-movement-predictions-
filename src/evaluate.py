from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds, average="weighted")
    }
