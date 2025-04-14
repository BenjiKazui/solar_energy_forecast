Example 1:
Training data 4 years: 2016, 2017, 2018, 2019
Test data 1 year: 2020

param_list = [("n_estimators", "int", 10, 200), ("learning_rate", "float", 0.01, 0.5), ("max_depth", "int", 3, 10), ("objective", "fixed", "reg:squarederror"),
              ("reg_lambda", "float", 0.0, 2.0), ("reg_alpha", "float", 0.0, 2.0), ("subsample", "fixed", 0.5), ("gamma", "float", 0.0, 2.0), ("verbosity", "fixed", 2)]
xgb_model, best_params, cv_scores, study = train_XGBoost(X_train=X_train, y_train=y_train, param_list=param_list, cv=3, scoring="neg_mean_absolute_error", n_trials=50, direction="minimize", random_state=random_state, save=False, save_path=None)
