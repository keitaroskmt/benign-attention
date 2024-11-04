import torch
from torch import Tensor


def add_label_noise(
    labels: Tensor,
    noise_ratio: float,
    num_classes: int,
    device: torch.device | str | int,
) -> Tensor:
    """
    Add label noise to a tensor of labels.
    Args:
        labels (Tensor): A tensor of labels.
        noise_ratio (float): The ratio of noisy labels.
        num_classes (int): The number of classes.
        device: The device to put the noisy labels on.
    Returns:
        Tensor: A tensor of noisy labels.
    """
    if noise_ratio <= 0:
        return labels
    mask = torch.rand_like(labels, dtype=torch.float) < noise_ratio
    noise_labels = (
        labels + torch.randint_like(labels, low=1, high=num_classes, device=device)
    ) % num_classes
    return torch.where(mask, noise_labels, labels)
