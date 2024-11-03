from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.datasets.noisy_dataset import NoisyDataset


def get_cifar10_datasets(
    noise_ratio: float = 0.0, size: int = 32, root: str = "~/pytorch_datasets"
) -> tuple[NoisyDataset, NoisyDataset]:
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    test_dataset = CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    train_dataset = NoisyDataset(
        train_dataset, noise_ratio=noise_ratio, key_target="targets"
    )
    test_dataset = NoisyDataset(
        test_dataset, noise_ratio=noise_ratio, key_target="targets"
    )
    return train_dataset, test_dataset
