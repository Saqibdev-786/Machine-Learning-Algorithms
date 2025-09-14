# Using SGDRegrssion

from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



df=pd.read_csv("Linear Regression/one Variable/archive/Salary_dataset.csv", index_col=0)

print(df)

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=SGDRegressor(max_iter=10000, learning_rate='constant', eta0=0.01)
model.fit(X_train,y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
w=model.coef_[0]
b=model.intercept_


print("\nModel Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Learned Coefficient (Slope): {w:.4f}")
print(f"Learned Intercept: {b}")
print(f"Equation of the line: y = {w:.4f}x + {b}")

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
plt.title('Linear Regression Model vs. Full Dataset')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.grid(True)
plt.show()
