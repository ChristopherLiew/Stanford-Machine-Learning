import numpy as np
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt

# Load our training data
data = pd.read_csv('/Users/MacBookPro15/Desktop/2020 Summer Coursera/Stanford ML/Programming Assignments in Python/ex2/ex2/ex2data1.txt', header=None)
X = data.iloc[:, [0,1]]
y = data.iloc[:, 2]

m = len(y)
bias_term = pd.Series(np.ones([m, 1]).flatten())

X_train = pd.concat([bias_term, X], axis=1).to_numpy()
y_train = y.to_numpy().reshape(m, 1)
theta = np.zeros((len(X.columns) + 1)) # for bgfs optimiser, theta.shape = (n,)

# Dataset 2
data2 = pd.read_csv('/Users/MacBookPro15/Desktop/2020 Summer Coursera/Stanford ML/Programming Assignments in Python/ex2/ex2/ex2data2.txt', header=None)

X2 = data.iloc[:, [0,1]]
y2 = data.iloc[:, 2]

m2 = len(y2)
bias_term_2 = pd.Series(np.ones([m2, 1]).flatten())
X_train_2 = pd.concat([bias_term_2, X2], axis=1).to_numpy()
y_train_2 = y2.to_numpy().reshape(m2, 1)
theta2 = np.zeros((len(X2.columns) + 1)) # for bgfs optimiser, theta.shape = (n,)

# Visualising our training data
pos = data.iloc[:, [0,1]][data.iloc[:, 2] == 1]
neg = data.iloc[:, [0,1]][data.iloc[:, 2] == 0]
plt.figure()
plt.scatter(pos.iloc[:, 0], pos.iloc[:, 1], c="b") # Postive = Blue
plt.scatter(neg.iloc[:, 0], neg.iloc[:, 1], c="r") # Negative = Red
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.show()

# Sigmoid function
def get_sigmoid(z):
    sig_val = 1/(1 + np.exp(-z))
    return sig_val

# Logistic Regression Cost Function with Regularisation
def compute_log_cost(theta, X, y, lmda = 0):
    m = len(y)
    n = len(theta)
    theta = theta.reshape(n, 1)
    hypothesis = get_sigmoid(np.dot(X, theta))
    J = (1/m) * sum(-y * np.log(hypothesis) - (1-y) * np.log(1 - hypothesis)) + (lmda/(2*m)) * sum(np.dot(theta.transpose(), theta))
    return J

# Test case (J should be approximately 0.693 for both datasets)
compute_log_cost(theta, X_train, y_train)
compute_log_cost(theta2, X_train_2, y_train_2)

# Gradient Function with Regularisation
def compute_log_gradient(theta, X, y, lmda = 0):
    m = len(y)
    n = len(theta)
    theta = theta.reshape(n, 1)
    hypothesis = get_sigmoid(np.dot(X, theta))
    grad = ((1/m) * sum((hypothesis - y) * X)) + (lmda/ m) * theta.reshape(n,)
    return grad

compute_log_gradient(theta, X_train, y_train)
compute_log_gradient(theta2, X_train_2, y_train_2)

# Using an optimiser instead of homemade GD
Result = op.minimize(fun=compute_log_cost,
                     x0=theta, # Must be of shape (n,)
                     args=(X_train, y_train),
                     method="TNC",
                     jac=compute_log_gradient);

optimal_theta = Result.x

# Predictions (X and theta must be numpy matrices of (m, n) & (n, ) respectively
def log_predict(X, theta = optimal_theta, threshold = 0.5):
    theta = theta.reshape(len(theta), 1)
    prob = get_sigmoid(np.dot(X, theta))
    results = prob > threshold
    return results.astype(int)

pred = log_predict(X_train, optimal_theta)

def classification_accuracy(pred, actual):
    compare = (pred == actual).astype(int)
    return sum(compare)/ len(pred)

classification_accuracy(pred, y_train) # 89% accuracy
