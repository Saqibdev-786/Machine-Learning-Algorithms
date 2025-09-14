# Implement Linear Regression of Multi variable using sklearn library

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

data=datasets.load_diabetes()

print("Feature Names: ", data.feature_names)
print("First 5 rows of data:\n", data.data[:5])
print("Corresponding labels:", data.target[:5])


X=data.data
y=data.target


print("\nFull Dataset Shapes:")
print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


print("\nAfter Train-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


model=LinearRegression()

model.fit(X_train, y_train)

predictions=model.predict(X_test)

print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("R^2 Score: ", r2_score(y_test, predictions))
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Coefficients:  [ -47.88303306 -253.52208312  457.16371972  352.02999697 -795.37064959
#   506.22154615  142.56380846  196.30002985  751.09154938  102.02650055]
# Intercept:  150.54870825321706
