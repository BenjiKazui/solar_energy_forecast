


def train_predict_evaluate(X, y, X_2, model_type, best_params):
    # drop time columns
    X_2_time = X_2["time"]
    X_2 = X_2.drop(columns=["time"])
    X = X.drop(columns=["time"])
    y = y.drop(columns=["time"])

    if model_type == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(**best_params)
    # implement the other models
    if model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**best_params)
    else:
        print("Use 'xgboost', 'linear_regression', .. or ..")
    model.fit(X, y)
    preds = model.predict(X_2)
    return model, preds, X_2_time