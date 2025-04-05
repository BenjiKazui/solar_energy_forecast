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
from src.do_plotting import plot_vertically

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

# STEP 6: Provide param_list for HPO, do HPO, find best parameters, train model with best parameters on entire training data
param_list = [("n_estimators", "int", 50, 200), ("learning_rate", "float", 0.01, 0.3), ("max_depth", "int", 3, 10), ("objective", "fixed", "reg:squarederror")]
xgb_model, best_params, cv_scores, study = train_XGBoost(X_train=X_train, y_train=y_train, param_list=param_list, cv=3, scoring="neg_mean_absolute_error", n_trials=2, direction="minimize", random_state=random_state, save=False, save_path=None)

lr_model = train_linear_regression(X_train, y_train, save=False, save_path=None)

# STEP 7.1: Predict on X_test using a baseline model and calculate metrics
lr_preds, lr_preds_df, lr_metrics_results = predict_evaluate(lr_model, X_test, y_test, metrics=["mae", "mse", "rmse"])

# STEP 7.2: Predict on X_test using the best_model and calculate metrics
xgb_preds, xgb_preds_df, xgb_metrics_results = predict_evaluate(xgb_model, X_test, y_test, metrics=["mae", "mse", "rmse"])

# STEP 8: Plot the data
# training data
plot_vertically(data_list=[y_train], label_list=["y_train"], window_size=24*30)
# test data and lr predictions
plot_vertically(data_list=[y_test, lr_preds_df[["time", "energy predictions"]]], label_list=["y_test", "preds"], window_size=24*30)
# test data and xgb predictions
plot_vertically(data_list=[y_test, xgb_preds_df[["time", "energy predictions"]]], label_list=["y_test", "preds"], window_size=24*30)

#print("X_train head\n", X_train.head())
#print("X_test head\n", X_test.head())

print("lr metrics results:\n", lr_metrics_results)
print("xgb metrics results:\n", xgb_metrics_results)

# 153.17876544161652
# mae: 156.77368488773968

# add baseline model
# maybe add another visualization