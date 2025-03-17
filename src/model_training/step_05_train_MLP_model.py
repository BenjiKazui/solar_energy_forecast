from src.step_03_data_preprocessing import data_preprocessing
from src.step_04_feature_engineering import create_time_based_features, create_lag_features, create_sun_position_features, create_interaction_features, create_fourier_features

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import joblib

import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Set max width to prevent wrapping
pd.set_option("display.max_colwidth", None)  # Don't truncate cell content


random.seed(42) # Fixes Python's built-in random module
np.random.seed(42) # Fixes NumPy's random behavior
tf.random.set_seed(42) # Fixes TensorFlow's internal randomness (weight initialization, dropout, etc.)

X, y = data_preprocessing()


# no added features: MAE 132.8821
# only added time_based_features: MAE 157
# only added sun_position_features: MAE 
# time + sun_position features: MAE 
# time + sun + lag: MAE 
# time + sun + lag + interaction: MAE 
# lag + interaction: MAE 
# only lag: MAE 
# lag + sun: MAE 
# only interaction: MAE 
# time + lag + interaction: MAE 
# lag + sun + interaction: MAE 108


#X = create_time_based_features(X)
#X = create_lag_features(X)
#X = create_sun_position_features(X)
#X = create_interaction_features(X)




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


scaler = StandardScaler()

def scale_certain_features(df_to_scale, cols_to_not_scale, scaler, fit_transform_or_transform):
    cols_to_transform = [col for col in df_to_scale.columns if col not in cols_to_not_scale]
    df_scaled = df_to_scale.copy()
    if fit_transform_or_transform == "fit_transform":
        df_scaled[cols_to_transform] = scaler.fit_transform(df_to_scale[cols_to_transform])
    elif fit_transform_or_transform == "transform":
        df_scaled[cols_to_transform] = scaler.transform(df_to_scale[cols_to_transform])
    else:
        print("Please specify whether to fit_transform or transform the data, use 'fit_transform' or 'transform' as the third argument")
        return None
    return df_scaled

X_train = scale_certain_features(X_train, ["hour_sin", "hour_cos"], scaler, "fit_transform")
X_test = scale_certain_features(X_test, ["hour_sin", "hour_cos"], scaler, "transform")

print("Number of Features: ", X_train.shape[1])

def build_model():
    # Define the MLP model
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1) # Output Layer
    ])

    # Compuile the model
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model

# cross validation
kf = KFold(n_splits=3, shuffle=False)
print("kf: ", kf)
fold = 1
cv_scores = []
for train_idx, val_idx in kf.split(X_train):
    print("a")
    print(train_idx, val_idx, fold)

    X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = build_model()

    history = model.fit(X_train_cv, y_train_cv, validation_data=(X_val_cv, y_val_cv), epochs=50, batch_size=32, verbose=1)

    val_loss, val_mae = model.evaluate(X_val_cv, y_val_cv, verbose=1)
    print(f"Fold: {fold}, Validation Loss: {val_loss}, Validation MAE: {val_mae}")
    
    cv_scores.append([val_loss, val_mae])

    fold += 1


print("Cross-validation MAE-scores:", cv_scores)
print("Mean MAE-score:", np.mean(cv_scores))


y_pred = model.predict(X_test)

# Evaluate performance
mae = model.evaluate(X_test, y_test)[1]
print("MAE: ", mae)


results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
#print(results)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

#plt.plot(results["y_test"], label="y_test")
#plt.plot(results["y_pred"], label="y_pred")
#plt.show()

#joblib.dump(model, "C:/Users/Brudo/solar_energy_forecast/models/xgboost_model.pkl")
