from src.step_03_data_preprocessing import data_preprocessing
from src.step_04_feature_engineering import create_time_based_features, create_lag_features, create_sun_position_features, create_interaction_features, create_fourier_features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Set max width to prevent wrapping
pd.set_option("display.max_colwidth", None)  # Don't truncate cell content


X, y = data_preprocessing()

# LOL using additionally time_based_features with the simple LinearRegressionModel gives a MAE of 220, not using them gives a MAE of 174.77
# no added features: MAE 174.77
# only added time_based_features: MAE 220
# only added sun_position_features: MAE 226
# time + sun_position features: MAE  266
# time + sun + lag: MAE 241
# time + sun + lag + interaction: MAE 232
# lag + interaction: MAE 173
# only lag: MAE 176
# only interaction: MAE 176
# time + lag + interaction: MAE 234
# lag + sun + interaction: MAE 215


#X = create_time_based_features(X)
X = create_lag_features(X)
#X = create_sun_position_features(X)
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

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)
print(y_pred)

mae = mae(y_test, y_pred)
print("MAE: ", mae)

results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
#print(results)

plt.plot(results["y_test"], label="y_test")
plt.plot(results["y_pred"], label="y_pred")
plt.show()



"""
# MAE = 174.77

from step_03_data_preprocessing import data_preprocessing
from step_04_feature_engineering import create_time_based_features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

import matplotlib.pyplot as plt

X, y = data_preprocessing()

X.drop(columns=["time"], inplace=True)
y.drop(columns=["time"], inplace=True)

#X = create_time_based_features(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

#print(X_train)
#print(y_train)

#print(X_test)
#print(y_test)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)
print(y_pred)

mae = mae(y_test, y_pred)
print("MAE: ", mae)

results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
print(results)

plt.plot(results["y_test"], label="y_test")
plt.plot(results["y_pred"], label="y_pred")
plt.show()
"""