import numpy as np
import torch
from torch import Tensor


def add_label_noise(
    labels: Tensor | np.ndarray | list,
    noise_ratio: float,
    num_classes: int,
) -> Tensor | np.ndarray | list:
    """
    Add label noise to a tensor of labels.
    Args:
        labels: A list of labels.
        noise_ratio (float): The ratio of noisy labels.
        num_classes (int): The number of classes.
    Returns:
        A list of noisy labels.
    """
    if noise_ratio <= 0:
        return labels

    if isinstance(labels, Tensor):
        mask = torch.rand_like(labels, dtype=torch.float) < noise_ratio
        noise_labels = (
            labels + torch.randint_like(labels, low=1, high=num_classes)
        ) % num_classes
        return torch.where(mask, noise_labels, labels)
    else:
        if not isinstance(labels, np.ndarray | list):
            raise ValueError("labels must be a list, numpy array, or PyTorch tensor.")
        if isinstance(labels, list):
            labels = np.array(labels)

        mask = np.random.rand(*labels.shape) < noise_ratio
        noise_labels = (
            labels + np.random.randint(1, num_classes, labels.shape)
        ) % num_classes
        ret = np.where(mask, noise_labels, labels)
        return ret.tolist() if isinstance(labels, list) else ret
