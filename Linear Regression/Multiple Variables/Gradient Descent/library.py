# Using SGDRegrssion

from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data=datasets.load_diabetes()

print("Feature Names: ", data.feature_names)
print("First 5 rows of data:\n", data.data[:5])
print("Corresponding labels:", data.target[:5])


X=data.data
y=data.target


print("\nFull Dataset Shapes:")
print(X.shape)
print(y.shape)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=SGDRegressor(max_iter=10000, learning_rate='constant', eta0=0.01)
model.fit(X_train,y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
w=model.coef_
b=model.intercept_


print("\nModel Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Learned Coefficient (Slope): {w}")
print(f"Learned Intercept: {b}")

