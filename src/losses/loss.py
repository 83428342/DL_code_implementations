import numpy as np

# SSE (sum of squares for error)

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# batch 데이터에 대한 CEE (cross entropy error)

def cross_entropy_error(y, t, one_hot_encoding=False):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot_encoding:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size # one_hot_encoding이 적용된 경우
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # one_hot_encoding이 적용되지 않은 경우

# gradient descent

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x