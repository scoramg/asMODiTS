import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.utils import resample

# Generate synthetic time series data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Proximity Forest parameters
num_trees = 10
max_depth = 5

# Create a list to store the decision trees
trees = []

for _ in range(num_trees):
    # Sample a subset of the data with replacement
    X_subsample, y_subsample = resample(X, y)
    
    # Create and fit a decision tree
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X_subsample, y_subsample)
    
    # Add the tree to the list
    trees.append(tree)

# Predict on a new time series instance
new_instance = np.array([[0.5]])  # Replace with your test data
predictions = []

for tree in trees:
    prediction = tree.predict(new_instance)
    predictions.append(prediction)

# Calculate the final prediction as the mean of all tree predictions
final_prediction = np.mean(predictions)

print(f"Predicted value: {final_prediction}")
