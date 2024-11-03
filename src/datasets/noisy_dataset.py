from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset


@dataclass
class DataInput:
    data: Tensor
    target: Tensor
    original_target: Tensor | None = None
    attention_mask: Tensor | None = None


def collate_fn(batch: list[DataInput]) -> DataInput:
    """
    Collate function for `torch.utils.data.DataLoader`
    """
    data = torch.stack([item.data for item in batch])
    target = torch.stack([item.target for item in batch])
    if any(item.original_target is None for item in batch):
        original_target = None
    else:
        original_target = torch.stack([item.original_target for item in batch])
    if any(item.attention_mask is None for item in batch):
        attention_mask = None
    else:
        attention_mask = torch.stack([item.attention_mask for item in batch])
    return DataInput(
        data=data,
        target=target,
        original_target=original_target,
        attention_mask=attention_mask,
    )


class NoisyDataset(TorchDataset):
    def __init__(
        self,
        dataset: TorchDataset | HFDataset,
        noise_ratio: float = 0.0,
        key_data: str = "data",
        key_target: str = "targets",
    ):
        """
        Dataset with noisy labels.
        `__getitem__` returns a tuple of (data, label, original_label)
        Args:
            dataset: Original dataset
            noise_ratio: Ratio of noisy labels
            key_data: Name of the data attribute
            key_target: Name of the target attribute
        """
        if isinstance(dataset, TorchDataset):
            if not hasattr(dataset, key_target):
                raise ValueError(f"Dataset does not have attribute {key_target}")
            # Make a deep copy of labels
            if isinstance(getattr(dataset, key_target), torch.Tensor):
                self.noisy_targets = getattr(dataset, key_target).detach().clone()
            elif isinstance(getattr(dataset, key_target), np.ndarray) or isinstance(
                getattr(dataset, key_target), list
            ):
                self.noisy_targets = torch.tensor(getattr(dataset, key_target))
            else:
                raise ValueError("Target must be a tensor, numpy array or list")
        elif isinstance(dataset, HFDataset):
            if key_data not in dataset.column_names:
                raise ValueError(f"Dataset does not have attribute {key_data}")
            if key_target not in dataset.column_names:
                raise ValueError(f"Dataset does not have attribute {key_target}")
            # Make a deep copy of labels
            if isinstance(dataset[key_target], torch.Tensor):
                self.noisy_targets = dataset[key_target].detach().clone()
            elif isinstance(dataset[key_target], np.ndarray) or isinstance(
                dataset[key_target], list
            ):
                self.noisy_targets = torch.tensor(dataset[key_target])
            else:
                raise ValueError("Target must be a tensor, numpy array or list")
        else:
            raise ValueError(
                "Dataset must be a `torch.utils.data.Dataset` or `datasets.Dataset`"
            )
        self.key_data = key_data
        self.key_target = key_target
        self.dataset = dataset

        self.noise_ratio = noise_ratio
        self.num_classes = len(set(self.noisy_targets))

        # Add label noise
        num_noisy = int(len(self.noisy_targets) * self.noise_ratio)
        noisy_indices = np.random.choice(
            len(self.noisy_targets), num_noisy, replace=False
        )

        for idx in noisy_indices:
            original_target = self.noisy_targets[idx]
            noisy_target = np.random.choice(
                [i for i in range(self.num_classes) if i != original_target]
            )
            assert self.noisy_targets[idx] != noisy_target
            self.noisy_targets[idx] = noisy_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> DataInput:
        attention_mask = None
        if isinstance(self.dataset, TorchDataset):
            data, original_target = self.dataset[index]
        elif isinstance(self.dataset, HFDataset):
            input = self.dataset[index]
            data = input[self.key_data]
            original_target = input[self.key_target]
            if "attention_mask" in self.dataset.column_names:
                attention_mask = input["attention_mask"]
        else:
            assert False, "Unreachable"
        target = self.noisy_targets[index]

        if not (original_target is None or isinstance(original_target, torch.Tensor)):
            original_target = torch.tensor(original_target)
        if not (attention_mask is None or isinstance(attention_mask, torch.Tensor)):
            attention_mask = torch.tensor(attention_mask)
        return DataInput(
            data=data if isinstance(data, torch.Tensor) else torch.tensor(data),
            target=target if isinstance(target, torch.Tensor) else torch.tensor(target),
            original_target=original_target,
            attention_mask=attention_mask,
        )
