import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.datasets.utils import add_label_noise


def get_cifar10_datasets(
    noise_ratio: float = 0.0,
    size: int = 32,
    root: str = "~/pytorch_datasets",
    use_transform: bool = True,
) -> tuple[Dataset, Dataset]:
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    if use_transform:
        # Note that we don't use RandomHorizontalFlip and RandomCrop to fix training dataset.
        # This is because our interest is on the overfitting behavior of the model.
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.ToTensor()

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)

    if noise_ratio > 0.0:
        train_dataset.targets = add_label_noise(
            train_dataset.targets, noise_ratio, num_classes=10
        )

    return train_dataset, test_dataset
