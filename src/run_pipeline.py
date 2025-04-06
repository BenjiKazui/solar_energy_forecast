# Using weather data from Freiburg im Breisgau (City in the state of Baden-Württemberg in Germany) and energy data from the entire state
# of Baden-Württemberg to train models. Then use weather data (that wasn't used for training) from Freiburg im Breisgau to let the models predict
# the solar energy generation on. Finally evaluate their prediction using the solar energy generation for that time period of the entire
# state of Baden-Württemberg. Meaning we use weather data from one location to predict the solar energy generation for an entire region,
# which is not too great, but still an approximation that's worth looking into. 

from src.data_pulling import pull_historical_weather_data, pull_historical_energy_data, load_local_data
from src.data_preprocessing import data_preprocessing
from src.feature_engineering import create_features
from src.model_training.train_XGBoost_model import train_XGBoost
from src.model_training.train_Linear_regression_model import train_linear_regression
from src.predict_and_evaluate import predict_evaluate
from src.do_plotting import plot_vertically, plot_zoomed_in_window

import random
import numpy as np

# STEP 1: Fix seeds to ensure reproducability
random_state = 42
random.seed(random_state) # Fixes Python's built-in random module
np.random.seed(random_state) # Fixes NumPy's random behavior

# STEP 2: Get training data
# Pull it from APIs:
hist_weather_data = pull_historical_weather_data(save=False, save_path=None, start_year=2017, end_year=2019)
hist_energy_data, _ = pull_historical_energy_data(save=False, save_path=None, start_year=2017, end_year=2019)
# Or load it from local machine:
#hist_weather_data = load_local_data(local_path="")
#hist_energy_data = load_local_data(local_path="")

# STEP 3: Get test data
# Pull it from APIs:
hist_weather_test_data = pull_historical_weather_data(save=False, save_path=None, start_year=2020, end_year=2020)
hist_energy_test_data, _ = pull_historical_energy_data(save=False, save_path=None, start_year=2020, end_year=2020)
# Or load it from local machine:
#hist_weather_test_data = load_local_data(local_path="")
#hist_energy_test_data = load_local_data(local_path="")

# STEP 4: Data preprocessing
X_train, y_train = data_preprocessing(hist_weather_data, hist_energy_data)
X_test, y_test = data_preprocessing(hist_weather_test_data, hist_energy_test_data)

# STEP 5: Create features
X_train = create_features(X_train, ["time_based", "lag", "sun_position", "interaction", "fourier"])
X_test = create_features(X_test, ["time_based", "lag", "sun_position", "interaction", "fourier"])

# STEP 6: Provide param_list for HPO, do HPO, and find best parameters. Metric to optimize for HPO is the mean of the cross-validation scores.
# Afterwards train the model with best hyperparameters on the entire training data and return that model.
# The param_list is a list of tuples, where each tuple contains the name of the parameter, the type of the parameter (int, float, fixed), and the range of values to search over.
param_list = [("n_estimators", "int", 50, 200), ("learning_rate", "float", 0.01, 0.3), ("max_depth", "int", 3, 10), ("objective", "fixed", "reg:squarederror")]
xgb_model, best_params, cv_scores, study = train_XGBoost(X_train=X_train, y_train=y_train, param_list=param_list, cv=3, scoring="neg_mean_absolute_error", n_trials=2, direction="minimize", random_state=random_state, save=False, save_path=None)

lr_model = train_linear_regression(X_train, y_train, save=False, save_path=None)

# STEP 7.1: Predict on X_test using a baseline model and calculate metrics
lr_preds, lr_preds_df, lr_metrics_results = predict_evaluate(lr_model, X_test, y_test, metrics=["mae", "mse", "rmse"])

# STEP 7.2: Predict on X_test using the best_model and calculate metrics
xgb_preds, xgb_preds_df, xgb_metrics_results = predict_evaluate(xgb_model, X_test, y_test, metrics=["mae", "mse", "rmse"])

# STEP 8: Plot the data
# training data
plot_vertically(data_type="train data", data_list=[y_train], label_list=["y_train"], window_size=24*30)
# test data and lr predictions
plot_vertically(data_type="model preds", data_list=[y_test, lr_preds_df[["time", "energy predictions"]], xgb_preds_df[["time", "energy predictions"]]], label_list=["y_test", "LR preds", "XGB preds"], window_size=24*30)
# test data and xgb predictions
#plot_vertically(data_type="XGB model preds", data_list=[y_test, xgb_preds_df[["time", "energy predictions"]]], label_list=["y_test", "preds"], window_size=24*30)

print(y_test.head())
print(lr_preds_df.head())
print(xgb_preds_df.head())

plot_zoomed_in_window(data_list=[y_test, lr_preds_df[["time", "energy predictions"]], xgb_preds_df[["time", "energy predictions"]]], label_list=["y_test", "LR preds", "XGB preds"], start_date="2020-08-01", end_date="2020-08-08")
plot_zoomed_in_window(data_list=[y_test, lr_preds_df[["time", "energy predictions"]], xgb_preds_df[["time", "energy predictions"]]], label_list=["y_test", "LR preds", "XGB preds"], start_date="2020-12-01", end_date="2020-12-08")


print("lr metrics results:\n", lr_metrics_results)
print("xgb metrics results:\n", xgb_metrics_results)

"""
lr metrics results:
 {'mae': 219.80862622609786, 'mse': 114054.97395619411, 'rmse': 337.7202599137252}
xgb metrics results:
 {'mae': 157.4977050841131, 'mse': 88441.23725631427, 'rmse': 297.3907148118688}
"""

# Plot vertically using a rolling mean with a big window size smoothes the data a lot and can potentially make us think the LR model is as good as or even better than the XGB model just by looking at the plot.
# But if we zoom in on the actual data (no rolling mean or whatsoever), we can see that the XGB model is actually better than the LR model. The LR model predicts quite a lot of negative values, which is not possible in reality.
# Also if we look at the metrics, we can see that the XGB model performs better than the LR model.