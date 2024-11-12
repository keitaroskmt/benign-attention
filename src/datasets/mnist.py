import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import ViTImageProcessor

from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)


# Following the experimental setup in the paper:
# https://openreview.net/attachment?id=pF8btdPVTL_&name=supplementary_material#page=36.36
def get_mnist_snr_datasets(
    noise_ratio: float = 0.0,
    snr: float = 1.0,
    root: str = "~/pytorch_datasets",
    use_transform: bool = True,
) -> tuple[Dataset, Dataset]:
    size = 28

    if use_transform:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        assert False, "Transform is required for MNIST-SNR dataset. This is because we add the gaussian noise to the images."

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    train_noise = torch.randn(len(train_dataset), size, size)
    train_noise[:, 5:23, 5:23] = 0.0
    test_noise = torch.randn(len(test_dataset), size, size)
    test_noise[:, 5:23, 5:23] = 0.0

    train_dataset.data = train_dataset.data * snr + train_noise.reshape(-1, size, size)
    test_dataset.data = test_dataset.data * snr + test_noise.reshape(-1, size, size)

    if noise_ratio > 0.0:
        train_dataset.targets = add_label_noise(
            train_dataset.targets, noise_ratio, num_classes=10
        )

    return train_dataset, test_dataset


def get_mnist_snr_hf_datasets(
    processor: ViTImageProcessor,
    noise_ratio: float = 0.0,
    snr: float = 1.0,
) -> tuple[HFDataset, HFDataset]:
    raw_datasets = load_dataset("mnist")
    raw_datasets = raw_datasets.rename_column("image", "pixel_values")

    def transform(example_batch):
        # Reshape images from 2D (H, W) to 3D (H, W, C) and convert to 3-channels
        # Also, normalize pixel values to [0, 1]
        images = []
        for image in example_batch["pixel_values"]:
            image = np.repeat(np.reshape(image, (28, 28, 1)) / 255, 3, axis=2)
            noise = np.repeat(np.random.randn(28, 28, 1), 3, axis=2)
            noise[5:23, 5:23, :] = 0.0
            image = image * snr + noise
            images.append((image - np.min(image)) / (np.max(image) - np.min(image)))
        inputs = processor(images, do_rescale=False, return_tensors="pt")
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

    noise_ratio = 0.0
    if noise_ratio > 0.0:
        train_dataset = train_dataset.map(
            lambda example: {
                "label": add_label_noise(example["label"], noise_ratio, num_classes=10)
            },
            features=train_dataset.features,
        )
    return train_dataset, test_dataset
