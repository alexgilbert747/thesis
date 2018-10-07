import numpy as np

# Define activation functions and their derivatives wrt inputs.
def softmax(a):
    output = np.exp(a - np.max(a, 1)[:, np.newaxis])  # use x-max(x) to avoid over/underflow
    row_sum = np.sum(output, 1)
    return output / row_sum[:, np.newaxis]

def relu(a):
    a[a < 0] = 0.
    return a

def d_relu(a):
    a[a < 0] = 0.
    a[a > 0] = 1.
    return a

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))