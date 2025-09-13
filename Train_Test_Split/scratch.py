# Implementation of Train Test Split from scratch

from sklearn import datasets
import random

def manual_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):

    """
    Splits arrays or matrices into random train and test subsets.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature dataset.
    y : array-like, shape (n_samples,)
        Target values.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Seed used by the random number generator.

    Returns:
    --------
    X_train, X_test, y_train, y_test : lists
        Split datasets for training and testing. 
    """

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size should be between 0 and 1")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if random_state is not None:
        random.seed(random_state)
    
    # Create a list of indices and shuffle them if required
    indices = list(range(len(X)))
    if shuffle:
        random.shuffle(indices)
    
    test_len = int(len(X) * test_size)

    test_indices = indices[:test_len]
    train_indices = indices[test_len:]

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


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
X_train, X_test, y_train, y_test=manual_train_test_split(X, y, test_size=0.2)


print("\nAfter Train-Test Split:")
print(f"X_train shape: ({len(X_train)}, {len(X_train[0])})")
print(f"X_test shape: ({len(X_test)}, {len(X_test[0])})")
print(f"y_train shape: ({len(y_train)},)")
print(f"y_test shape: ({len(y_test)},)")



# Demonstrating effect of shuffling and random_state
X = [[i] for i in range(10)]
# Labels: first 5 samples labeled 0, next 5 labeled 1
y = [0]*5 + [1]*5

# Split WITHOUT shuffling: Use this to understand shuffle and random_state
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.3, shuffle=True, random_state=None)

print("\nExample Split (with shuffle and random_state):")
print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)

