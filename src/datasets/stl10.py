import logging

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import ViTImageProcessor

from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)


# Used in the experiment in `main_vit_finetune.py`
def get_stl10_hf_datasets_for_finetune(
    processor: ViTImageProcessor,
    pretrain_sample_size: int,
    sample_size: int,
    noise_ratio: float = 0.0,
) -> tuple[HFDataset, HFDataset, HFDataset]:
    raw_datasets = load_dataset("tanganke/stl10")
    raw_datasets = raw_datasets.rename_column("image", "pixel_values")

    def transform(example_batch):
        inputs = processor(example_batch["pixel_values"], return_tensors="pt")
        inputs["label"] = torch.tensor(example_batch["label"])
        return inputs

    raw_datasets.set_transform(transform)
    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    assert pretrain_sample_size + sample_size <= len(
        train_dataset
    ), "Sample size is too large."

    random_indices = np.random.choice(
        len(train_dataset), pretrain_sample_size + sample_size, replace=False
    )
    pretrain_dataset = train_dataset.select(random_indices[:pretrain_sample_size])
    finetune_dataset = train_dataset.select(random_indices[pretrain_sample_size:])

    logger.info(f"Key names in the dataset: {train_dataset.column_names}.")
    logger.info(
        f"Types of input and target are : {type(train_dataset[0]['pixel_values']), type(train_dataset[0]['label'])}."
    )
    logger.info(f"Size of input tensor is : {train_dataset[0]['pixel_values'].shape}.")

    # Add label noise only to the finetune_dataset
    noise_ratio = 0.0
    if noise_ratio > 0.0:
        finetune_dataset = finetune_dataset.map(
            lambda example: {
                "label": add_label_noise(example["label"], noise_ratio, num_classes=10)
            },
            features=finetune_dataset.features,
        )
    return pretrain_dataset, finetune_dataset, test_dataset
