from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
import pandas as pd
import numpy as np

df = pd.read_csv("Linear Regression/one Variable/archive/Salary_dataset.csv")
df=df.iloc[:, 1:].reset_index(drop=True)

print(df)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# Show sample data
print("First 5 samples:")
for i in range(5):
    print(f"Feature: {X[i][0]}, Label: {y[i]}")


# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Learned Coefficient (Slope): {model.coef_[0]:.4f}")
print(f"Learned Intercept: {model.intercept_:.4f}")


print(model.predict(np.array([[6]]))[0])

