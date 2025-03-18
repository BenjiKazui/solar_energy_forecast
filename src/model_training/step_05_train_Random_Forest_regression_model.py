from src.step_03_data_preprocessing import data_preprocessing
from src.step_04_feature_engineering import create_time_based_features, create_lag_features, create_sun_position_features, create_interaction_features, create_fourier_features

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Set max width to prevent wrapping
pd.set_option("display.max_colwidth", None)  # Don't truncate cell content


X, y = data_preprocessing()


# no added features: MAE 144
# only added time_based_features: MAE 165
# only added sun_position_features: MAE 107
# time + sun_position features: MAE 116
# time + sun + lag: MAE 116
# time + sun + lag + interaction: MAE 115
# lag + interaction: MAE 134
# only lag: MAE 135
# only interaction: MAE 144
# time + lag + interaction: MAE 149
# lag + sun + interaction: MAE 106 -> 107 (n_estimators=50) -> 106 (n_estimators=120) -> 106 (n_estimators=200)


#X = create_time_based_features(X)
X = create_lag_features(X)
X = create_sun_position_features(X)
X = create_interaction_features(X)




#X[["solar_elevation", "solar_azimuth"]].to_csv("C:/Users/Brudo/Desktop/solar_parameters_quick_check.csv", index=False)

#X.to_csv("C:/Users/Brudo/Desktop/all_parameters_quick_check.csv", index=True)

X.drop(columns=["time"], inplace=True)
y.drop(columns=["time"], inplace=True)

print("X isna SUM")
print(X.isna().sum())
for col in X.columns:
    for index, value in X[col].items():
        if pd.isna(value):
            print(f"Index {index} in column {col} is NULL")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

#print(X_train)
#print(y_train)

#print(X_test)
#print(y_test)

# changing alpha doenst change the MAE a lot
model = RandomForestRegressor(n_estimators=100, random_state=42)

# do GridSearch
param_grid = {"n_estimators": [60, 80, 100, 120, 140, 160]}
grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search.fit(X_train, y_train.values.flatten())
cv_results_grid_search = pd.DataFrame(grid_search.cv_results_)

# do RandomSearch
param_dist = {"n_estimators": np.arange(60, 160, 10)}
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train.values.flatten())
cv_results_random_search = pd.DataFrame(random_search.cv_results_)

print("Results grid_search:\n", cv_results_grid_search)
print("Results random_search:\n", cv_results_random_search)

cv_results_grid_search = cv_results_grid_search.sort_values(by="mean_test_score", ascending=False)
print("ranking of all models (grid search):\n", cv_results_grid_search[['params', 'rank_test_score']].sort_values(by='rank_test_score'))
cv_results_random_search = cv_results_random_search.sort_values(by="mean_test_score", ascending=False)
print("ranking of all models (random search):\n", cv_results_random_search[['params', 'rank_test_score']].sort_values(by='rank_test_score'))



print("y_train.shape:\n", y_train.shape)

# Finally fit the model. Cross-Validation returns the model with the best hyperparameters from cross-validation, trained on the entire training set
# use the best model from random_search
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
#print(y_pred)

mae = mae(y_test, y_pred)
print("MAE: ", mae)

results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
#print(results)

plt.plot(results["y_test"], label="y_test")
plt.plot(results["y_pred"], label="y_pred")
plt.show()

#joblib.dump(model, "C:/Users/Brudo/solar_energy_forecast/models/Random_Forest_regression_model.pkl")
