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

import random
import numpy as np

random.seed(42) # Fixes Python's built-in random module
np.random.seed(42) # Fixes NumPy's random behavior

hist_weather_data = pull_historical_weather_data(save=False, save_path=None, start_year=2019, end_year=2019)
hist_energy_data, _ = pull_historical_energy_data(save=False, save_path=None, start_year=2019, end_year=2019)

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
y = y.drop(columns=["time"])

print(X)

param_list = [("n_estimators", "int", 50, 200), ("learning_rate", "float", 0.01, 0.3), ("max_depth", "int", 3, 10), ("objective", "fixed", "reg:squarederror")]

best_model, mae, cv_scores, study = train_XGBoost(X=X, y=y, test_size=0.2, param_list=param_list, cv=3, scoring="neg_mean_absolute_error", n_trials=100, direction="minimize", save=False, save_path=None)

print("mae: ", mae)
print("cv_scores: ", cv_scores)

# NEXT TO DO: implement feature_engineering and training of a model in the pipeline
# AFTER: implement evaluating a model using test data or probably cross validation
# AFTER: implement proper pulling of weather forecast
# AFTER: predict on weather forecast and save predictions
# AFTER/LATER: check how close predictions are to actual energy production during the predicted interval of time