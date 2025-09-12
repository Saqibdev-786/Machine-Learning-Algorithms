from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 rows of data:\n", iris.data[:5])
print("Corresponding labels:", iris.target[:5])


#split it in features and labels
X=iris.data
y=iris.target

print(X.shape)
print(y.shape)


#hours of study vs good/bad grades
#10different students
#train with 8
#predict with the remaining data
#allows for determining the model accuracy
#level of accuracy

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X = np.arange(10).reshape((10, 1))
# Labels: first 5 samples labeled 0, next 5 labeled 1
y = np.array([0]*5 + [1]*5)

# Split WITHOUT shuffling: Use this to understand shuffle and random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=None)

print("Training labels:", y_train)
print("Testing labels:", y_test)
