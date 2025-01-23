import torch
from torchvision import datasets, transforms
import os

def load_mnist(data_dir=None, flatten=True):
    """
    MNIST 데이터셋을 로드하고 반환
    :param data_dir: 데이터를 저장할 경로. None이면 기본 경로 사용.
    :param flatten: True면 데이터를 flatten (28x28 -> 784) 형태로 반환.
    :return: 훈련 데이터와 테스트 데이터 텐서
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "mnist_data")

    mnist_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    mnist_test = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # 데이터 분리
    x_train = torch.stack([data[0].squeeze() for data in mnist_train])
    t_train = torch.tensor([data[1] for data in mnist_train])

    x_test = torch.stack([data[0].squeeze() for data in mnist_test])
    t_test = torch.tensor([data[1] for data in mnist_test])

    # Flatten 처리
    if flatten:
        x_train = x_train.view(x_train.size(0), -1)  # 28x28 -> 784
        x_test = x_test.view(x_test.size(0), -1)

    return (x_train, t_train), (x_test, t_test)
