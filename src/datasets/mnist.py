import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


# Following the experimental setup in the paper:
# https://openreview.net/attachment?id=pF8btdPVTL_&name=supplementary_material#page=36.36
def get_mnist_snr_datasets(
    snr: float = 1.0, root: str = "~/pytorch_datasets"
) -> tuple[Dataset, Dataset]:
    size = 28

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    train_noise = torch.randn(len(train_dataset), size, size)
    train_noise[:, 5:23, 5:23] = 0.0
    test_noise = torch.randn(len(test_dataset), size, size)
    test_noise[:, 5:23, 5:23] = 0.0

    train_dataset.data = train_dataset.data * snr + train_noise.reshape(-1, size, size)
    test_dataset.data = test_dataset.data * snr + test_noise.reshape(-1, size, size)

    return train_dataset, test_dataset
