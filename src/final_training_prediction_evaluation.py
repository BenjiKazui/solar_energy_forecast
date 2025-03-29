


def train_predict_evaluate(X, y, X_test, model_type, best_params):
    import pandas as pd
    # drop time columns
    X_test_time = X_test["time"]
    X_test = X_test.drop(columns=["time"])
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
    preds = model.predict(X_test)
    final_preds_df = pd.DataFrame(data={"time": X_test_time, "energy": preds})
    return model, preds, X_test_time, final_preds_df