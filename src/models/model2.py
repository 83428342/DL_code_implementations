from ..data.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist()

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)