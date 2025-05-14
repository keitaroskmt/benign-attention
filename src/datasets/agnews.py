# http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
# https://huggingface.co/datasets/fancyzhx/ag_news
import logging
import random

import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from src.distributed_utils import main_process_first
from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)


def get_agnews_datasets(
    sample_size: int | None = None,
    noise_ratio: float = 0.0,
    model_name_or_path: str | None = "bert-base-uncased",
    pad_to_max_length: bool = True,
) -> tuple[Dataset, Dataset]:
    raw_datasets = load_dataset("fancyzhx/ag_news", cache_dir="~/pytorch_datasets")
    max_seq_length = 128
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    if model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        raise NotImplementedError("Only model_name_or_path is supported for now.")

    # Padding strategy
    if pad_to_max_length:
        padding = "max_length"
    else:
        padding = "longest"

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

    with main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, desc="Running tokenizer on dataset"
        )
        raw_datasets = raw_datasets.remove_columns(["text", "token_type_ids"])
        raw_datasets.set_format(type="torch")

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    if sample_size is not None:
        random_indices = np.random.choice(
            len(train_dataset), sample_size, replace=False
        )
        train_dataset = train_dataset.select(random_indices)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Number of classes in the dataset: {num_labels}.")
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # https://github.com/huggingface/datasets/issues/4684#issuecomment-1185696240
    if noise_ratio > 0.0:
        train_dataset = train_dataset.map(
            lambda example: {
                "label": add_label_noise(
                    example["label"], noise_ratio, num_classes=num_labels
                )
            },
            features=train_dataset.features,
        )

    return train_dataset, test_dataset


# Used in the experiment in `main_bert_finetune.py`
def get_agnews_datasets_for_finetune(
    pretrain_sample_size: int,
    sample_size: int,
    noise_ratio: float = 0.0,
    model_name_or_path: str | None = "bert-base-uncased",
    pad_to_max_length: bool = True,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    raw_datasets = load_dataset("fancyzhx/ag_news", cache_dir="~/pytorch_datasets")
    max_seq_length = 128
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    if model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        raise NotImplementedError("Only model_name_or_path is supported for now.")

    # Padding strategy
    if pad_to_max_length:
        padding = "max_length"
    else:
        padding = "longest"

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

    with main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, desc="Running tokenizer on dataset"
        )
        raw_datasets = raw_datasets.remove_columns(["text", "token_type_ids"])
        raw_datasets.set_format(type="torch")

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    assert pretrain_sample_size + sample_size <= len(
        train_dataset
    ), "Sample size is too large."

    np.random.seed(seed)
    random_indices = np.random.choice(
        len(train_dataset), pretrain_sample_size + sample_size, replace=False
    )
    pretrain_dataset = train_dataset.select(random_indices[:pretrain_sample_size])
    finetune_dataset = train_dataset.select(random_indices[pretrain_sample_size:])

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Number of classes in the dataset: {num_labels}.")
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Add label noise only to the finetune_dataset
    if noise_ratio > 0.0:
        finetune_dataset = finetune_dataset.map(
            lambda example: {
                "label": add_label_noise(
                    example["label"], noise_ratio, num_classes=num_labels
                )
            },
            features=finetune_dataset.features,
        )
    return pretrain_dataset, finetune_dataset, test_dataset
