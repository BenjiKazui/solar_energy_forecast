import pandas as pd
import numpy as np


def smape(y_test, preds):
    """
    Calculate symmetric mean absolute percentage error and using a small value (1e-10) to add to the denominator to avoid division by zero
    """
    return 100 / len(y_test) * np.sum(2 * np.abs(preds - y_test) / (np.abs(y_test) + np.abs(preds) + 1e-10))


def predict_evaluate(model, X_test, y_test, metrics):
    """
    Predict on X_test using the provided and trained model.
    Afterwards calculate metrics.
    """
    y_test = y_test.drop(columns=["time"])
    
    y_test = np.array(y_test).flatten()

    preds = model.predict(X_test.drop(columns=["time"]))
    preds = np.array(preds).flatten()
    assert preds.shape == y_test.shape, f"Shape mismatch: preds {preds.shape} vs y_test {y_test.shape}."

    preds_df = pd.DataFrame(data={"time": X_test["time"], "energy predictions": preds, "true energy generation": y_test})

    metrics_results = {}

    for metric in metrics:
        if metric == "mae":
            from sklearn.metrics import mean_absolute_error
            metrics_results["mae"] = mean_absolute_error(y_test, preds)
        #elif metric == "smape":
        #    metrics_results["smape"] = smape(y_test, preds)
        elif metric == "mse":
            from sklearn.metrics import mean_squared_error
            metrics_results["mse"] = mean_squared_error(y_test, preds)
        elif metric == "rmse":
            from sklearn.metrics import root_mean_squared_error
            metrics_results["rmse"] = root_mean_squared_error(y_test, preds)

    return preds, preds_df, metrics_results