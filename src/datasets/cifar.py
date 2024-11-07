from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_cifar10_datasets(
    size: int = 32, root: str = "~/pytorch_datasets"
) -> tuple[Dataset, Dataset]:
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
    return train_dataset, test_dataset
