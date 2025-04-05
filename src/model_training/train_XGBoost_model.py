import xgboost as xgb
import joblib
from sklearn.model_selection import cross_val_score
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


def train_XGBoost(X_train, y_train, param_list, cv, scoring, n_trials, direction, random_state, save=False, save_path=None):
    """
    Does a HPO with XGBoost models using the provided param_list.
    Within the HPO a cross validation is done. The mean of all mean_average_errors from each fold of the cv is used to optimize the study, thus to find the best model.
    After the HPO a model with the best found hyperparameters is newly initialized and trained on ALL training data.
    Only argument 'minimize' implemented for parameter 'direction'.
    """
    # drop time columns, because we don't want to use it in that format, instead we are using the features 'hour', 'day_of_year' and 'month'.
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
    best_model = xgb.XGBRegressor(**study.best_params, random_state=random_state)
    best_model.fit(X_train, y_train)

    if save == True and save_path != None:
        joblib.dump(best_model, save_path)
        print("XGBoost Model saved to: ", save_path)

    return best_model, best_params, cv_scores, study