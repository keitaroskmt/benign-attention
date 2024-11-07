from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_cifar10_datasets(
    size: int = 32, root: str = "~/pytorch_datasets"
) -> tuple[Dataset, Dataset]:
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    # Note that we don't use RandomHorizontalFlip and RandomCrop to fix training dataset.
    # This is because our interest is on the overfitting behavior of the model.
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset
