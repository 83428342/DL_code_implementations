import numpy as np

# step function

def step_function(x):
    y = x > 0
    return y.astype(int) # boolean 값의 0, 1 반환

# sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# relu

def relu(x):
    return np.maximum(0, x)

# identity function

def identity_function(x):
    return x

# softmax

def softmax(a):
    c = np.max(a) # overflow 방지
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y