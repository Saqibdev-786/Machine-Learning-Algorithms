# Not using Built in Gradient Descent

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class LinearRegressionGD:
    def __init__(self, num_iters=1000, learning_rate=0.01):
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.n_iterations = num_iters
    
    def compute_cost(self, X, y):
        n = len(y)
        y_pred = self.coef_ * X + self.intercept_
        cost = (1/(2*n)) * np.sum((y_pred - y) ** 2)
        return cost

    def compute_gradients(self, X, y):
        n = len(y)
        y_pred = X.dot(self.coef_) + self.intercept_
        dw = (1/n) * np.dot(X.T, (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)
        return dw, db

    def fit(self, X, y):

        print(X.shape)
        
        X=X.flatten()
        
        print(X.shape)

        self.coef_ = 0
        self.intercept_ = 0

        for i in range(self.n_iterations):
            dw, db = self.compute_gradients(X, y)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            if i % math.ceil(self.n_iterations / 10) == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}: Cost {cost}")

    def predict(self, X):
        return self.coef_*X + self.intercept_

# Load dataset
df=pd.read_csv("Linear Regression/one Variable/archive/Salary_dataset.csv", index_col=0)
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model=LinearRegressionGD(num_iters=10000, learning_rate=0.01)
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
print(f"R-squared (R2): {r2:.4f}")
print(f"Weight (w): {w:.4f}")
print(f"Bias (b): {b:.4f}")


# 1. Create a range of x-values that spans the entire dataset
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# 2. Use the trained model to predict y-values for this full range
y_line = model.predict(x_range)

# 3. Create the plot
plt.figure(figsize=(10, 6))

# Plot the entire dataset as a scatter plot
plt.scatter(X, y, color='blue', label='Data Points')

# Plot the linear regression line over the full range
plt.plot(x_range, y_line, color='red', linewidth=2, label='Linear Regression Line')

# Add labels, title, and legend
plt.title('Linear Regression with Gradient Descent (Single Variable)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

#Weight (w): 9423.8153
#Bias (b): 24380.2015
