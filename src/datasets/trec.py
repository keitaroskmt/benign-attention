# http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
# https://huggingface.co/datasets/fancyzhx/ag_news
import logging
import random

import numpy as np
from transformers import AutoTokenizer

from datasets import Dataset, load_dataset
from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)


def get_trec_datasets_for_finetune(  # noqa: PLR0913
    pretrain_sample_size: int,
    sample_size: int,
    noise_ratio: float = 0.0,
    model_name_or_path: str | None = "bert-base-uncased",
    pad_to_max_length: bool = True,  # noqa: FBT001, FBT002
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    raw_datasets = load_dataset(
        "trec",
        cache_dir="~/pytorch_datasets",
        trust_remote_code=True,
    )
    raw_datasets = raw_datasets.rename_column("coarse_label", "label")

    max_seq_length = 128
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    if model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        msg = "Only model_name_or_path is supported for now."
        raise NotImplementedError(msg)

    # Padding strategy
    padding = "max_length" if pad_to_max_length else "longest"

    def preprocess_function(examples):  # noqa: ANN001, ANN202
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    raw_datasets = raw_datasets.remove_columns(["text", "token_type_ids"])
    raw_datasets.set_format(type="torch")

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

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Number of classes in the dataset: {num_labels}.")  # noqa: G004
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")  # noqa: G004

    # Add label noise only to the finetune_dataset
    if noise_ratio > 0.0:
        finetune_dataset = finetune_dataset.map(
            lambda example: {
                "label": add_label_noise(
                    example["label"],
                    noise_ratio,
                    num_classes=num_labels,
                ),
            },
            features=finetune_dataset.features,
        )
    return pretrain_dataset, finetune_dataset, test_dataset
