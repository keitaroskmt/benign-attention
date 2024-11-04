# http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
# https://huggingface.co/datasets/fancyzhx/ag_news
import logging
import random

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from src.distributed_utils import main_process_first

logger = logging.getLogger(__name__)


def get_agnews_datasets(
    model_name_or_path: str | None = "bert-base-uncased",
    pad_to_max_length: bool = True,
) -> tuple[Dataset, Dataset]:
    raw_datasets = load_dataset("fancyzhx/ag_news", cache_dir="~/pytorch_datasets")
    max_seq_length = 128

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
        raw_datasets.set_format(type="torch")

    train_dataset = raw_datasets["train"]
    test_dataset = raw_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    return train_dataset, test_dataset
