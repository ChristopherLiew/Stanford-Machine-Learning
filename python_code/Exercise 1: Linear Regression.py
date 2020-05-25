import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load our training data
data = pd.read_csv('/Users/MacBookPro15/Desktop/2020 Summer Coursera/Stanford ML/Programming Assignments in Python/ex1/ex1/ex1data1.txt', header=None)
X = data.iloc[:, 0]
y = data.iloc[:, 1]
m = len(X)

# Plot our data
plt.figure()
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.scatter(X, y, c="r")
plt.show()

# Building our Linear Regression model with Gradient Descent
# Add bias term to our X variable(s), i.e. theta 0
bias_term = pd.Series(np.ones([m, 1]).flatten())
X = pd.concat([bias_term, X], axis=1).to_numpy()
y = y.to_numpy().reshape(m,1)
# Instantiate initial Theta
theta = np.array([0, 0]).reshape((2, 1)) # 2 by 1

# 1. Cost Function (Mean Squared Error)
def compute_cost(X, y, theta):
    # Compute Cost
    cost = (1/(2*m)) * sum((np.dot(X, theta) - y)**2)
    return cost[0]

compute_cost(X, y, theta) # Test if approximately 32.07

# 2. The Gradient Descent Function
def gradient_descent(X, y, init_theta, alpha = 0.01, num_iter = 100):
    J_log = []
    m = len(y)
    results = {}
    for i in range(num_iter):
        init_theta = init_theta - (alpha/m) * sum((np.dot(X, init_theta) - y) * X).reshape((len(init_theta), 1))
        J = compute_cost(X, y, init_theta)
        J_log.append(J)
    results["theta"] = init_theta
    results["J_val"] = J_log
    return results

results = gradient_descent(X, y, theta)

# Feature Normalisation (ie sklearns's Standard Scaler; For quicker convergence)
def standardise(X):
    mu = np.mean(X)
    std = np.std(X)
    norm = (X - mu)/std
    return norm

X_stand = standardise(X)

# Run Gradient Descent on X_stand
stand_results = gradient_descent(X_stand, y, theta)

# Plot results of Cost against Iterations for ex1data2
# Preprocess data2
data2 = pd.read_csv('/Users/MacBookPro15/Desktop/2020 Summer Coursera/Stanford ML/Programming Assignments in Python/ex1/ex1/ex1data2.txt', header=None)
X2 = data2.iloc[:, [0,1]]
y2 = data2.iloc[:, 2]
m2 = len(y2)

# Add bias term to our X variable(s), i.e. theta 0
bias_term_2 = pd.Series(np.ones([m2, 1]).flatten())
X2 = pd.concat([bias_term_2, X2], axis=1).to_numpy()
y2 = y2.to_numpy().reshape(m2,1)
# Instantiate initial Theta
theta2 = np.array([0, 0, 0]).reshape((3, 1)) # 2 by 1
# Standardise results
X2_stand = standardise(X2)
# Run GD on X2 Stand
results2 = gradient_descent(X2_stand, y2, theta2, num_iter=100)

# Plot
J_log = results2['J_val']
iterations = np.arange(1, len(J_log) + 1)
plt.figure()
plt.plot(iterations, J_log, marker = ".")
plt.show()

# Using the Normal Equation to find optimal theta
# Only if (X'X)^-1 is invertible else no unique solution
# Takes in X & y as numpy arrays
def normal_eqn(X, y):
    theta = np.zeros((X2.shape[1], 1))
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    return theta

normal_eqn(X2, y2)
