# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
import os
import logging
import random
import contextlib

import torch.distributed as dist
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from src.datasets.noisy_dataset import NoisyDataset

logger = logging.getLogger(__name__)


# https://github.com/huggingface/transformers/blob/004530aa050efcdd489f1ac6809626fa578636ad/src/transformers/training_args.py#L2446
@contextlib.contextmanager
def main_process_first(local=True, desc="work"):
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.

    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    Args:
        local (`bool`, *optional*, defaults to `True`):
            if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
            rank 0 In multi-node environment with a shared filesystem you most likely will want to use
            `local=False` so that only the main process of the first node will do the processing. If however, the
            filesystem is not shared, then the main process of each node will need to do the processing, which is
            the default behavior.
        desc (`str`, *optional*, defaults to `"work"`):
            a work description to be used in debug logs
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        main_process_desc = "main local process" if local else "main procecs"
        is_main_process = local_rank == 0 if local else rank == 0

        try:
            if not is_main_process:
                # tell all replicas to wait
                logger.debug(
                    f"{rank}: waiting for the {main_process_desc} to perform {desc}"
                )
                dist.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(
                    f"{rank}: {main_process_desc} completed {desc}, releasing all replicas"
                )
                dist.barrier()
    else:
        yield


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
    noise_ratio: float = 0.0,
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

    train_dataset = NoisyDataset(
        train_dataset,
        noise_ratio=noise_ratio,
        key_data="input_ids",
        key_target="label",
    )
    valid_dataset = NoisyDataset(
        valid_dataset,
        noise_ratio=noise_ratio,
        key_data="input_ids",
        key_target="label",
    )
    test_dataset = NoisyDataset(
        test_dataset,
        noise_ratio=noise_ratio,
        key_data="input_ids",
        key_target="label",
    )

    return train_dataset, valid_dataset, test_dataset
