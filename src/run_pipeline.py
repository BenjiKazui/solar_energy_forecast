#import random
#import numpy as np
#from sklearn.model_selection import train_test_split
#from src.data_preprocessing import data_preprocessing
#from src.feature_engineering import create_features
#from src.train_XGBoost_model import train_model


from src.data_pulling import pull_historical_weather_data, pull_historical_energy_data, pull_future_weather_data
#from data_pulling import load_local_hist_weather_data, load_local_hist_energy_data, load_local_future_weather_data
from src.data_preprocessing import data_preprocessing_2
from src.feature_engineering import create_features
from src.model_training.train_XGBoost_model import train_XGBoost
from src.final_training_prediction_evaluation import train_predict_evaluate

import random
import numpy as np
import pandas as pd

random.seed(42) # Fixes Python's built-in random module
np.random.seed(42) # Fixes NumPy's random behavior

hist_weather_data = pull_historical_weather_data(save=False, save_path=None, start_year=2019, end_year=2020)
hist_energy_data, _ = pull_historical_energy_data(save=False, save_path=None, start_year=2019, end_year=2020)

future_weather_data = pull_future_weather_data(save=False, save_path=None)

#print("hist_weather_data: ", hist_weather_data)


X, y = data_preprocessing_2(hist_weather_data, hist_energy_data)

# check for NaNs/missing values in X and y
#print(X.describe())
#print(y.describe())
#X_nan_counts = X.isna().sum()
#y_nan_counts = y.isna().sum()
#print("X_nan_counts: ", X_nan_counts)
#print("y_nan_counts: ", y_nan_counts)

X = create_features(X, ["time_based", "lag", "sun_position", "interaction"])

#print("X after feature creation\n", X)
#print("X.describe()\n", X.describe())

param_list = [("n_estimators", "int", 50, 200), ("learning_rate", "float", 0.01, 0.3), ("max_depth", "int", 3, 10), ("objective", "fixed", "reg:squarederror")]

best_model, best_params, y_pred, mae, cv_scores, study, X_time = train_XGBoost(X=X, y=y, test_size=0.2, param_list=param_list, cv=3, scoring="neg_mean_absolute_error", n_trials=20, direction="minimize", save=False, save_path=None)

print("mae: ", mae)
print("cv_scores: ", cv_scores)

print(future_weather_data.describe())
future_weather_data = future_weather_data.rename(columns={"shortwave_radiation": "G(i)", "temperature_2m": "T2m", "wind_speed_10m": "WS10m"})
#future_weather_data.drop(columns=["time"], inplace=True)
#print("future_weather_data:\n", future_weather_data)
X_2 = future_weather_data.copy()

X_2["time"] = pd.to_datetime(X_2["time"])

X_2 = create_features(X_2, ["time_based", "lag", "sun_position", "interaction"])

print("X_2\n", X_2)
print(X_2.dtypes)

final_model, final_preds, X_2_time = train_predict_evaluate(X, y, X_2, "xgboost", best_params)

#print("final_preds:\n", final_preds)

print("X_time:\n", X_time)
print("X_2_time:\n", X_2_time)

# plot x-axis datetime (make sure to handle the leap year)
# plot y-axis the actual data y and the prediction, use different color for each year

#print(X_2.describe())
# NEXT TO DO: implement feature_engineering and training of a model in the pipeline
# AFTER: implement evaluating a model using test data or probably cross validation
# AFTER: implement proper pulling of weather forecast
# AFTER: predict on weather forecast and save predictions
# AFTER/LATER: check how close predictions are to actual energy production during the predicted interval of time