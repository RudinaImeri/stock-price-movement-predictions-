from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )
    model.fit(X, y)
    return model
