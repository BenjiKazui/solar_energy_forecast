import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import optuna

def get_params(trial, param_list):
    """
    Function to extract the parameters with their respective parameter range from the param_list.
    Each trial of the HPO calls this function to pick values of the parameters for that specific trial.
    """

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


def train_XGBoost(X_train, y_train, param_list, cv, scoring, n_trials, direction, save=False, save_path=None):
    """
    Only argument 'minimize' implemented for parameter 'direction'.
    """
    # drop time columns, because we don't want to use it in that format, instead we are using the features 'hour', 'day_of_year' and 'month'.
    X_time = X_train["time"]
    X_train = X_train.drop(columns=["time"])
    y_train = y_train.drop(columns=["time"])

    # keep track of cv-scores
    cv_scores = []

    # define objective function to optimize with optuna
    def objective(trial):

        # get parameter values for each trial
        params = get_params(trial, param_list)
        # build the model
        model = xgb.XGBRegressor(**params, random_state=42)

        # do cross validation and get the respective scores. Uses negative mean absolute error as a metric.
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        # take the negative mean of all cv scores to get a positive mean absolute error as the metric for optuna to optimize on. 
        score = -score.mean()
        cv_scores.append(score)

        return score

    # Create study object and give it information about the direction minimize MAE
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