import torch
from torchvision import datasets, transforms

def load_mnist(data_dir="mnist_data"):
    """
    MNIST 데이터셋을 로드하고 반환합
    :param data_dir: 데이터를 저장할 경로
    :return: 훈련 데이터와 테스트 데이터 텐서
    """
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

    return (x_train, t_train), (x_test, t_test)
