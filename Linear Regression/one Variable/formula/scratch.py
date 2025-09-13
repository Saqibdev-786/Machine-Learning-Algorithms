# Implement Linear Regression of Multi variable using only pure Python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionforOneVariable:
    def __init__(self):
        # Initialize the model's parameters to None.
        self.coef_ = None  # Stores the slope (m) of the line.
        self.intercept_ = None  # Stores the y-intercept (c) of the line.
    
    def fit(self, X, y):
        """
        Calculates the optimal slope and y-intercept for a one-variable
        linear regression model using pure Python.
        """
        # Ensure X and y are simple lists, flattening them if necessary.
        X_flat = [item for sublist in X for item in sublist]
        y_flat = y

        # Calculate the means of X and y using pure Python.
        X_mean = sum(X_flat) / len(X_flat)
        y_mean = sum(y_flat) / len(y_flat)

        # Initialize variables for the numerator and denominator of the slope formula.
        numerator = 0
        denominator = 0
        
        # Iterate through the data to calculate the sums for the slope formula.
        for i in range(len(X_flat)):
            numerator += (X_flat[i] - X_mean) * (y_flat[i] - y_mean)
            denominator += (X_flat[i] - X_mean)**2

        # Check for a zero denominator to prevent division by zero.
        if denominator == 0:
            self.coef_ = 0
            self.intercept_ = y_mean
        else:
            # Calculate the slope (m).
            self.coef_ = numerator / denominator
            # Calculate the y-intercept (c).
            self.intercept_ = y_mean - self.coef_ * X_mean
    
    def predict(self, X):
        """
        Predicts the target values using the learned slope and intercept.
        Accepts a single number or a list of numbers.
        """
        if isinstance(X, (int, float)):
            # If the input is a single number, return a single prediction.
            return self.coef_ * X + self.intercept_
        else:
            # If the input is a list, iterate and return a list of predictions.
            predictions = []
            for x_val in X:
                predictions.append(self.coef_ * x_val[0] + self.intercept_)
            return predictions

# -----------------------------------------------------------------------------

# Load the dataset using pandas.
df = pd.read_csv("Linear Regression/one Variable/archive/Salary_dataset.csv")
# Remove the first column (likely an unnamed index) and reset the index.
df = df.iloc[:, 1:].reset_index(drop=True)

# Print the full dataframe for verification.
print("Full DataFrame:")
print(df.to_string(index=False))

# Select features (X) and the target (y) and convert them to Python lists.
X = df.iloc[:, :-1].values.tolist()
y = df.iloc[:, -1].values.tolist()

# Show sample data.
print("\nFirst 5 samples:")
for i in range(5):
    # Access the single feature value at index 0.
    print(f"Feature: {X[i][0]}, Label: {y[i]}")

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of our Linear Regression model.
model = LinearRegressionforOneVariable()
# Train the model on the training data.
model.fit(X_train, y_train)

# Make predictions on the test set.
predictions = model.predict(X_test)

# Print the learned parameters of the model.
print("\nLearned Model Parameters:")
print(f"Learned Slope (m): {model.coef_}")
print(f"Learned Y-intercept (c): {model.intercept_}")

# Evaluate the model's performance.
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

r2 = r2_score(y_test, predictions)
print(f"R-squared Score: {r2:.4f}")

# Make a single prediction for 6 years of experience.
predicted_salary = model.predict(6)
print(f"Predicted salary for 6 years of experience: {predicted_salary:.2f}")