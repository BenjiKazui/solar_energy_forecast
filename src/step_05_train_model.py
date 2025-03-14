from step_03_data_preprocessing import data_preprocessing
from step_04_feature_engineering import create_time_based_features

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd

import matplotlib.pyplot as plt

X, y = data_preprocessing()

# LOL using additionally time_based_features with the simple LinearRegressionModel gives a MAE of 220, not using them gives a MAE of 174.77
X = create_time_based_features(X)

X.drop(columns=["time"], inplace=True)
y.drop(columns=["time"], inplace=True)

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