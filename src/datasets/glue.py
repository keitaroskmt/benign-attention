# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
import logging
import random

import numpy as np
from transformers import AutoTokenizer

from datasets import Dataset, load_dataset
from src.datasets.utils import add_label_noise

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_glue_datasets_for_finetune(  # noqa: PLR0913
    pretrain_sample_size: int,
    sample_size: int,
    noise_ratio: float = 0.0,
    task_name: str = "sst2",
    model_name_or_path: str | None = "bert-base-uncased",
    pad_to_max_length: bool = True,  # noqa: FBT001, FBT002
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    if task_name not in task_to_keys:
        msg = f"Task {task_name} not found in GLUE tasks."
        raise ValueError(msg)
    if task_name != "sst2":
        msg = "Only SST-2 is supported for now."
        raise NotImplementedError(msg)

    raw_datasets = load_dataset(
        "nyu-mll/glue",
        task_name,
        cache_dir="~/pytorch_datasets",
    )
    max_seq_length = 128
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    if model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        msg = "Only model_name_or_path is supported for now."
        raise NotImplementedError(msg)

    sentence_1_key, sentence_2_key = task_to_keys[task_name]

    # Padding strategy
    padding = "max_length" if pad_to_max_length else "longest"

    label2id = {label: id_ for id_, label in enumerate(label_list)}
    id2label = {id_: label for label, id_ in label2id.items()}

    def preprocess_function(examples):  # noqa: ANN001, ANN202
        args = (
            (examples[sentence_1_key],)
            if sentence_2_key is None
            else (examples[sentence_1_key], examples[sentence_2_key])
        )
        return tokenizer(
            *args,
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
    if task_name == "sst2":
        raw_datasets = raw_datasets.remove_columns(
            ["sentence", "token_type_ids", "idx"],
        )
    raw_datasets.set_format(type="torch")

    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets[
        "validation_matched" if task_name == "mnli" else "validation"
    ]
    test_dataset = raw_datasets["test_matched" if task_name == "mnli" else "test"]

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

    # https://github.com/huggingface/datasets/issues/4684#issuecomment-1185696240
    if task_name == "sst2" and noise_ratio > 0.0:
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
    return pretrain_dataset, finetune_dataset, valid_dataset
