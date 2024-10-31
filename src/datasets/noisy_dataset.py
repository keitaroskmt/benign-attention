import numpy as np
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from copy import deepcopy

from wandb.sdk.data_types.image import TorchTensorType


class NoisyDataset(TorchDataset, HFDataset):
    def __init__(
        self,
        dataset: TorchDataset | HFDataset,
        noise_ratio: float = 0.0,
        key_input: str = "data",
        key_target: str = "targets",
    ):
        """
        Dataset with noisy labels.
        `__getitem__` returns a tuple of (data, label, original_label)
        Args:
            dataset: Original dataset
            noise_ratio: Ratio of noisy labels
            key_input: Name of the input attribute
            key_target: Name of the target attribute
        """
        if isinstance(dataset, TorchDataset):
            if not hasattr(dataset, key_input):
                raise ValueError(f"Dataset does not have attribute {key_input}")
            if not hasattr(dataset, key_target):
                raise ValueError(f"Dataset does not have attribute {key_target}")
            self.input = getattr(dataset, key_input)
            self.target = getattr(dataset, key_target)
        elif isinstance(dataset, HFDataset):
            if key_input not in dataset.column_names:
                raise ValueError(f"Dataset does not have attribute {key_input}")
            if key_target not in dataset.column_names:
                raise ValueError(f"Dataset does not have attribute {key_target}")
            self.input = dataset[key_input]
            self.target = dataset[key_target]
        else:
            raise ValueError(
                "Dataset must be either `torch.utils.data.Dataset` or `datasets.Dataset`"
            )

        self.noise_ratio = noise_ratio
        self.noisy_targets = deepcopy(self.target)
        self.num_classes = len(set(self.target))

        # Add label noise
        num_noisy = int(len(self.noisy_targets) * self.noise_ratio)
        noisy_indices = np.random.choice(
            len(self.noisy_targets), num_noisy, replace=False
        )

        for idx in noisy_indices:
            original_target = self.target[idx]
            noisy_target = np.random.choice(
                [i for i in range(self.num_classes) if i != original_target]
            )
            assert self.noisy_targets[idx] != noisy_target
            self.noisy_targets[idx] = noisy_target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        input = self.input[index]
        original_target = self.target[index]
        target = self.noisy_targets[index]
        return input, target, original_target
