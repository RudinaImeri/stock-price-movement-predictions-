from sklearn.ensemble import RandomForestClassifier


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=15,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model
