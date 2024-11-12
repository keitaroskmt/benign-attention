import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import ViTImageProcessor

from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)


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


def get_cifar10_hf_datasets(
    processor: ViTImageProcessor,
    noise_ratio: float = 0.0,
) -> tuple[HFDataset, HFDataset]:
    raw_datasets = load_dataset("cifar10")
    raw_datasets = raw_datasets.rename_column("img", "pixel_values")

    def transform(example_batch):
        inputs = processor(example_batch["pixel_values"], return_tensors="pt")
        inputs["label"] = torch.tensor(example_batch["label"])
        return inputs

    raw_datasets.set_transform(transform)

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    logger.info(f"Key names in the dataset: {train_dataset.column_names}.")
    logger.info(
        f"Types of input and target are : {type(train_dataset[0]["pixel_values"]), type(train_dataset[0]["label"])}."
    )
    logger.info(f"Size of input tensor is : {train_dataset[0]['pixel_values'].shape}.")

    if noise_ratio > 0.0:
        train_dataset = train_dataset.map(
            lambda example: {
                "label": add_label_noise(example["label"], noise_ratio, num_classes=10)
            },
            features=train_dataset.features,
        )
    return train_dataset, test_dataset
