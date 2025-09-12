from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris=datasets.load_iris()

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 rows of data:\n", iris.data[:5])
print("Corresponding labels:", iris.target[:5])


#split it in features and labels
X=iris.data
y=iris.target

print("\nFull Dataset Shapes:")
print(X.shape)
print(y.shape)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


print("\nAfter Train-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



X = np.arange(10).reshape((10, 1))
# Labels: first 5 samples labeled 0, next 5 labeled 1
y = np.array([0]*5 + [1]*5)

# Split WITHOUT shuffling: Use this to understand shuffle and random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=None)

print("\nExample Split (with shuffle and random_state):")
print("X_train:", X_train.flatten())
print("y_train:", y_train)
print("X_test:", X_test.flatten())
print("y_test:", y_test)
