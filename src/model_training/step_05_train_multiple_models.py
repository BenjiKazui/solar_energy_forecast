



# With this file one can train many models (and do an automated HPO?)

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


user_input = input("Which model do you want to train? (1: LinearRegression, 2: LogisticRegression, 3: Lasso, 4: Ridge, 5: DecisionTreeRegressor, 6: RandomForestRegressor, all: all models)\n")

model_numbers = user_input.replace(",", " ").replace(";", " ").split()

print(model_numbers)

for number in model_numbers:
    number = int(number)
    print(number)