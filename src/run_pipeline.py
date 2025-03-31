
# Using weather data from Freiburg im Breisgau (City in the state of Baden-Württemberg in Germany) and energy data from the entire state
# of Baden-Württemberg to train models. Then use weather data from Freiburg im Breisgau from a different time period to let the models predict
# the solar energy generation on. Finally evaluate their prediction using the solar energy generation for that time period of the entire
# state of Baden-Württemberg. Meaning we use weather data from one location to predict the solar energy generation for an entire region,
# which is not too great, but still an approximation that's worth looking into. 


from src.data_pulling import pull_historical_weather_data, pull_historical_energy_data, pull_future_weather_data, load_local_data
from src.data_preprocessing import data_preprocessing
from src.feature_engineering import create_features
from src.model_training.train_XGBoost_model import train_XGBoost
from src.final_training_prediction_evaluation import train_predict_evaluate
from src.do_plotting import plot_vertically

import random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

# STEP 1: Fix seeds to ensure reproducability
random.seed(42) # Fixes Python's built-in random module
np.random.seed(42) # Fixes NumPy's random behavior

# STEP 2: Get training data
# Pull it from APIs:
hist_weather_data = pull_historical_weather_data(save=False, save_path=None, start_year=2018, end_year=2019)
hist_energy_data, _ = pull_historical_energy_data(save=False, save_path=None, start_year=2018, end_year=2019)
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

X, y = data_preprocessing(hist_weather_data, hist_energy_data)
X_test, y_test = data_preprocessing(hist_weather_test_data, hist_energy_test_data)

print("X_test:\n", X_test)
print("y_test:\n", y_test)

X = create_features(X, ["time_based", "lag", "sun_position", "interaction"])

param_list = [("n_estimators", "int", 50, 200), ("learning_rate", "float", 0.01, 0.3), ("max_depth", "int", 3, 10), ("objective", "fixed", "reg:squarederror")]

best_model, best_params, cv_scores, study, X_time = train_XGBoost(X=X, y=y, param_list=param_list, cv=3, scoring="neg_mean_absolute_error", n_trials=2, direction="minimize", save=False, save_path=None)

print("cv_scores: ", cv_scores)

X_test["time"] = pd.to_datetime(X_test["time"])

X_test = create_features(X_test, ["time_based", "lag", "sun_position", "interaction"])

# I only need this if test_size > 0.0 in "train_XGBoost" to train on the entire training data
final_model, final_preds, X_test_time, final_preds_df = train_predict_evaluate(X, y, X_test, "xgboost", best_params)

print("111 y_test:\n", y_test)
print("111 y_test.mean:\n", y_test["energy"].mean())
print("111 final_preds:\n", final_preds)
print("111 final_preds.mean:\n", final_preds.mean())


plot_vertically([y_test, final_preds_df], ["y_test", "preds"])

mae_test = mean_absolute_error(y_test.drop(columns=["time"]), final_preds)

print("mae_test:\n", mae_test)

# 156.1685791015625
# 154.84197998046875
# 159.86949157714844
# 156.22654724121094
# 155.06646728515625
# 155.55967712402344

### CLEAN UP
### Implement another model
### GET READY TO PUT GITHUB LINK TO PROJECT IN MY CURRICULUM VITAE





## Visualization, whats the goal? Sanity checking our predictions? Visualizing just the historical data or predictions aswell?
## What to do with the rolling of the historical data vs the predicted data, where the predicted data has only data of several days

### Try to use thinner lines
### implement using final_preds and y together



#print(X_2.describe())
# NEXT TO DO: implement feature_engineering and training of a model in the pipeline
# AFTER: implement evaluating a model using test data or probably cross validation
# AFTER: implement proper pulling of weather forecast
# AFTER: predict on weather forecast and save predictions
# AFTER/LATER: check how close predictions are to actual energy production during the predicted interval of time