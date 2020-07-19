import numpy as np
import math

def hypothesis(x, theta):
    """
      x is training set
      theta is weight parameter
      """
    return np.transpose(np.array(theta)).dot(np.array(x))

def costFunction(theta, x, y):
    """
        x is training set => (j, 2)
        y is a vector of
        theta is weight parameter
        """
    factor = 1 / 2
    sum = 0
    for i in range(0, len(x)):
        sum += math.pow((hypothesis(x[i], theta) - y[i]), 2)
    return factor * sum

def learnThetaSingle(theta, x, y, alpha):
    return theta + alpha * (y - hypothesis(x, theta)) * x

def learnTheta(theta, x, y, alpha):
    f = theta
    for i in range(0, len(x)):
        f = learnThetaSingle(f, x[i], y[i], alpha)
    return f

testX = np.array([[1, 2],
                   [4, 6],
                   [5, 123],
                   [41, -14],
                   [-413, 0],
                   [0, 0],
                   [5, 12],
                   [-3, -14],
                   [1, 1004],
                   [51, 51]])

testY = np.array([3, 10, 128, 27, -413, 0, 17, -17, 1005, 102])
theta = [2, 3]
print(costFunction(theta, testX, testY)) 
theta = learnTheta(theta, testX, testY, 0.0001)
print(theta)
