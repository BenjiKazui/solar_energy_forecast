from src.step_03_data_preprocessing import data_preprocessing
from src.step_04_feature_engineering import create_time_based_features, create_lag_features, create_sun_position_features, create_interaction_features, create_fourier_features

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import make_scorer
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import random
import numpy as np

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Set max width to prevent wrapping
pd.set_option("display.max_colwidth", None)  # Don't truncate cell content


X, y = data_preprocessing()


# no added features: MAE 141
# only added time_based_features: MAE 179
# only added sun_position_features: MAE 106
# time + sun_position features: MAE 
# time + sun + lag: MAE 
# time + sun + lag + interaction: MAE 133
# lag + interaction: MAE 
# only lag: MAE 135
# lag + sun: MAE 107
# only interaction: MAE 141
# time + lag + interaction: MAE 
# lag + sun + interaction: MAE 87


#X = create_time_based_features(X)
X = create_lag_features(X)
X = create_sun_position_features(X)
X = create_interaction_features(X)

random.seed(42) # Fixes Python's built-in random module
np.random.seed(42) # Fixes NumPy's random behavior



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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#print(X_train)
#print(y_train)

#print(X_test)
#print(y_test)

# changing alpha doenst change the MAE a lot
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="reg:squarederror"
)

kf = KFold(n_splits=3, shuffle=False)

mae_scorer = make_scorer(mae, greater_is_better=False)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=mae_scorer)
print("Cross-validation R2-scores:", scores)
print("Mean R2-score:", scores.mean())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

mae = mae(y_test, y_pred)
print("MAE: ", mae)

results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
#print(results)

plt.plot(results["y_test"], label="y_test")
plt.plot(results["y_pred"], label="y_pred")
plt.show()

#joblib.dump(model, "C:/Users/Brudo/solar_energy_forecast/models/xgboost_model.pkl")
