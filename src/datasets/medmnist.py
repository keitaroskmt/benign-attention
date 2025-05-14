import logging

import numpy as np
import torch
from transformers import ViTImageProcessor

from datasets import Dataset as HFDataset
from datasets import load_dataset
from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)


def get_medmnist_hf_datasets_for_finetune(  # noqa: PLR0913
    processor: ViTImageProcessor,
    pretrain_sample_size: int,
    sample_size: int,
    noise_ratio: float = 0.0,
    task_name: str = "pneumoniamnist",
    seed: int = 42,
) -> tuple[HFDataset, HFDataset, HFDataset]:
    if task_name not in ["pneumoniamnist", "breastmnist"]:
        msg = f"Task {task_name} is not supported."
        raise NotImplementedError(msg)
    num_classes = 2
    raw_datasets = load_dataset("albertvillanova/medmnist-v2", task_name)
    raw_datasets = raw_datasets.rename_column("image", "pixel_values")

    def transform(example_batch) -> dict[str, torch.Tensor]:  # noqa: ANN001
        # Reshape images from 2D (H, W) to 3D (H, W, C) and convert to 3-channels
        # Also, normalize pixel values to [0, 1]
        images = []
        for image in example_batch["pixel_values"]:
            image = np.repeat(np.reshape(image, (28, 28, 1)), 3, axis=2)  # noqa: PLW2901
            images.append(image)
        inputs = processor(images, return_tensors="pt")
        inputs["label"] = torch.tensor(example_batch["label"])
        return inputs

    raw_datasets.set_transform(transform)
    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    if pretrain_sample_size + sample_size > len(train_dataset):
        msg = "Sample size is too large."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    random_indices = rng.choice(
        len(train_dataset),
        pretrain_sample_size + sample_size,
        replace=False,
    )
    pretrain_dataset = train_dataset.select(random_indices[:pretrain_sample_size])
    finetune_dataset = train_dataset.select(random_indices[pretrain_sample_size:])

    logger.info(f"Key names in the dataset: {train_dataset.column_names}.")  # noqa: G004
    logger.info(
        f"Types of input and target are : {type(train_dataset[0]['pixel_values']), type(train_dataset[0]['label'])}.",  # noqa: E501, G004
    )
    logger.info(f"Size of input tensor is : {train_dataset[0]['pixel_values'].shape}.")  # noqa: G004

    # Add label noise only to the finetune_dataset
    if noise_ratio > 0.0:
        finetune_dataset = finetune_dataset.map(
            lambda example: {
                "label": add_label_noise(
                    example["label"],
                    noise_ratio,
                    num_classes=num_classes,
                ),
            },
            features=finetune_dataset.features,
        )
    return pretrain_dataset, finetune_dataset, test_dataset
