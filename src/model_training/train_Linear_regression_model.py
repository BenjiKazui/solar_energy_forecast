from sklearn.linear_model import LinearRegression
import joblib

def train_linear_regression(X_train, y_train, save=False, save_path=None):
    """
    Train a linear regression model using the training data.
    """

    # Drop the 'time' column from the training data
    X_train = X_train.drop(columns=["time"])
    y_train = y_train.drop(columns=["time"])

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    if save == True and save_path != None:
        joblib.dump(model, save_path)
        print("Linear Regression Model saved to: ", save_path)

    return model