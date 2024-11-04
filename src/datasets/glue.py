# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
import logging
import random

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from src.distributed_utils import main_process_first

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


def get_glue_datasets(
    task_name: str = "sst2",
    model_name_or_path: str | None = "bert-base-uncased",
    pad_to_max_length: bool = True,
) -> tuple[Dataset, Dataset, Dataset]:
    if task_name not in task_to_keys:
        raise ValueError(f"Task {task_name} not found in GLUE tasks.")
    if task_name != "sst2":
        raise NotImplementedError("Only SST-2 is supported for now.")

    raw_datasets = load_dataset(
        "nyu-mll/glue", task_name, cache_dir="~/pytorch_datasets"
    )
    max_seq_length = 128
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    if model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        raise NotImplementedError("Only model_name_or_path is supported for now.")

    sentence_1_key, sentence_2_key = task_to_keys[task_name]

    # Padding strategy
    if pad_to_max_length:
        padding = "max_length"
    else:
        padding = "longest"

    label2id = {label: id for id, label in enumerate(label_list)}
    id2label = {id: label for label, id in label2id.items()}

    def preprocess_function(examples):
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

    with main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function, batched=True, desc="Running tokenizer on dataset"
        )
        raw_datasets.set_format(type="torch")

    train_dataset = raw_datasets["train"]
    valid_dataset = raw_datasets[
        "validation_matched" if task_name == "mnli" else "validation"
    ]
    test_dataset = raw_datasets["test_matched" if task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    return train_dataset, valid_dataset, test_dataset
