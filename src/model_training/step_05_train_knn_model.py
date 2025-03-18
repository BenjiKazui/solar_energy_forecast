from src.step_03_data_preprocessing import data_preprocessing
from src.step_04_feature_engineering import create_time_based_features, create_lag_features, create_sun_position_features, create_interaction_features, create_fourier_features

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


# no added features: MAE 143
# only added time_based_features: MAE 145
# only added sun_position_features: MAE 
# time + sun_position features: MAE 
# time + sun + lag: MAE 
# time + sun + lag + interaction: MAE 
# lag + interaction: MAE 
# only lag: MAE 
# only interaction: MAE 
# time + lag + interaction: MAE 
# lag + sun + interaction: MAE 133 (n_neighbors=5) -> 131 (n_neighbors=10) -> 129 (n_neighbors=15) -> 129 (n_neighbors=20)


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

# do RandomizedSearchCV to find the best hyperparameters
model = KNeighborsRegressor(n_neighbors=20)
param_distributions = {"n_neighbors": np.arange(5, 30 ,2)}
random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=10, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# get the best model from the randomized search
best_model = random_search.best_estimator_

results = random_search.cv_results_
results = pd.DataFrame(results)
print(results)

y_pred = best_model.predict(X_test)
#print(y_pred)

mae = mae(y_test, y_pred)
print("MAE: ", mae)

results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
#print(results)

plt.plot(results["y_test"], label="y_test")
plt.plot(results["y_pred"], label="y_pred")
plt.show()

#joblib.dump(best_model, "C:/Users/Brudo/solar_energy_forecast/models/knn_model.pkl")
