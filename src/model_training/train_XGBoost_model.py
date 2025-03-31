"""
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
import optuna

#pd.set_option("display.max_rows", None)  # Show all rows
#pd.set_option("display.max_columns", None)  # Show all columns
#pd.set_option("display.width", 1000)  # Set max width to prevent wrapping
#pd.set_option("display.max_colwidth", None)  # Don't truncate cell content


X, y = data_preprocessing()


#X = create_time_based_features(X)
X = create_lag_features(X)
X = create_sun_position_features(X)
X = create_interaction_features(X)

random.seed(42) # Fixes Python's built-in random module
np.random.seed(42) # Fixes NumPy's random behavior


X.drop(columns=["time"], inplace=True)
y.drop(columns=["time"], inplace=True)

print("X isna SUM")
print(X.isna().sum())
for col in X.columns:
    for index, value in X[col].items():
        if pd.isna(value):
            print(f"Index {index} in column {col} is NULL")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)



def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "objective": "reg:squarederror"
    }

    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_absolute_error")
    score = -score.mean()

    return score

# minimize MAE
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best parameters:\n", study.best_params)

# Train model with the best found hyperparameters on the whole training set
best_model = xgb.XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)
#print(y_pred)

mae = mae(y_test, y_pred)
print("MAE: ", mae)

results = pd.DataFrame(data={"y_test": y_test["energy_generated"], "y_pred": y_pred.flatten()})
#print(results)

plt.plot(results["y_test"], label="y_test")
plt.plot(results["y_pred"], label="y_pred")
plt.show()

#joblib.dump(best_model, "C:/Users/Brudo/solar_energy_forecast/models/xgboost_model.pkl")
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import random
import numpy as np
import optuna

def get_params(trial, param_list):
    param_dict = {}

    for param in param_list:
        if len(param) == 4:
            name, param_type, low, high = param
            if param_type == "int":
                param_dict[name] = trial.suggest_int(name, low, high)
            elif param_type == "float":
                param_dict[name] = trial.suggest_float(name, low, high)
            else:
                print("Use 'int', 'float', or 'fixed' for the param_type")
        elif len(param) == 3:
            name, param_type, value = param
            if param_type == "fixed":
                param_dict[name] = value
            else:
                print("Use 'int', 'float', or 'fixed' for the param_type")
        else:
            print("Only parameter ranges for length of 3 or length of 4 are implented. Meaning (name, param_type, low, high) or (name, param_type='fixed', value)")

    return param_dict


def train_XGBoost(X, y, test_size, param_list, cv, scoring, n_trials, direction, save=False, save_path=None):
    # drop time columns
    X_time = X["time"]
    X = X.drop(columns=["time"])
    y = y.drop(columns=["time"])

    # split the data
    if test_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    else:
        X_train = X.copy()
        y_train = y.copy()
        X_test = None
        y_test = None
    # keep track of cv-scores
    cv_scores = []

    # define objective function to optimize with optuna
    def objective(trial):

        params = get_params(trial, param_list)
        model = xgb.XGBRegressor(**params, random_state=42)

        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        print("score: ", score)
        score = -score.mean()
        cv_scores.append(score)

        return score

    # minimize MAE
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    # Get the best_params for final_training of the model (another script, see final_training_prediction_evaluation)
    print("Best parameters:\n", study.best_params)
    best_params = study.best_params

    # Train model with the best found hyperparameters on the whole training set
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    if test_size > 0.0:
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        print("MAE: ", mae)

        results = pd.DataFrame(data={"y_test": y_test["energy"], "y_pred": y_pred.flatten()})
        #print(results)

        plt.plot(results["y_test"], label="y_test")
        plt.plot(results["y_pred"], label="y_pred")
        plt.show()
    
        if save == True and save_path != None:
            joblib.dump(best_model, "C:/Users/Brudo/solar_energy_forecast/models/xgboost_model.pkl")

        return best_model, best_params, y_pred, mae, cv_scores, study, X_time
    
    else:
        if save == True and save_path != None:
            joblib.dump(best_model, "C:/Users/Brudo/solar_energy_forecast/models/xgboost_model.pkl")

        return best_model, best_params, cv_scores, study, X_time