# Implement Linear Regression of Multi variable using only numpy

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error

class LinearRegressionforOneVariable:
    def __init__(self):
        self.coef_=None
        self.intercept_=None
    
    def fit(self, X, y):
        
        # Ensure X is a 1-D array to handle potential column vectors.
        X = X.flatten()
        # Ensure y is a 1-D array.
        y = y.flatten()

        X_mean=np.mean(X)
        y_mean=np.mean(y)

        m=X.shape[0]

        num=np.sum((X-X_mean)*(y-y_mean))
        den=np.sum((X-X_mean)**2)

        # Check for a zero denominator to prevent division by zero errors.
        # This occurs if all X values are identical.
        if den == 0:
            # If denominator is zero, the slope is 0 (a horizontal line).
            self.coef_ = 0
            # The y-intercept is simply the mean of y.
            self.intercept_ = y_mean
        else:
            # Calculate the slope (m) using the formula: covariance(X, y) / variance(X).
            self.coef_ = num / den
            # Calculate the y-intercept (c) using the formula: c = y_mean - m * x_mean.
            self.intercept_ = y_mean - self.coef_ * X_mean
    

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            np.array(X)
        
        return self.coef_*X+self.intercept_



# Generate synthetic data for one variable linear regression
df = pd.read_csv("Linear Regression/one Variable/archive/Salary_dataset.csv")
df=df.iloc[:, 1:].reset_index(drop=True)

print(df)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# Show sample data
print("First 5 samples:")
for i in range(5):
    print(f"Feature: {X[i][0]}, Label: {y[i]}")


# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of our LinearRegression model.
model = LinearRegressionforOneVariable()
# Train the model on the training data.
model.fit(X_train, y_train)

# Make predictions on the test set.
predictions = model.predict(X_test)

# Print the learned parameters of the model.
print(f"Learned Slope (m): {model.coef_}")
print(f"Learned Y-intercept (c): {model.intercept_}")

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2:.4f}")

print(model.predict(6))
