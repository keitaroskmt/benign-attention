import os
import logging

import torch
from torch import nn, Tensor
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.nn.init import zeros_, orthogonal_, uniform_
from torch.nn.parameter import Parameter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.models.vit import ViT
from src.datasets.cifar import get_cifar10_datasets
from src.distributed_utils import setup, cleanup


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device | str | int
) -> tuple[Tensor, Tensor]:
    """
    Calculate the number of correct predictions and the total number of samples.
    """
    model.eval()
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            correct += pred.argmax(dim=1).eq(target).sum()
            total += len(target)
    return correct, total


class OurAttention(nn.Module):
    """
    Attention model in the setting of our paper.
    """

    def __init__(
        self, embed_dim: int, num_classes: int, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.W = Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
        self.p = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))
        self.nu = Parameter(torch.empty(embed_dim, num_classes, **factory_kwargs))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        orthogonal_(self.W)
        zeros_(self.p)
        uniform_(self.nu, -1 / self.embed_dim**0.5, 1 / self.embed_dim**0.5)

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return the output tensor (batch_size, num_classes) and the attention scores (batch_size, seq_len).
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
        """
        assert x.dim() == 3, f"Input must have 3 dimensions, got {x.dim()}"
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding dimension {embed_dim} must match layer embedding dimension {self.embed_dim}"

        attention_logits = (x @ self.W @ self.p).squeeze(-1)
        attention_scores = torch.softmax(attention_logits, dim=-1)

        token_scores = (x @ self.nu).squeeze(-1)
        output = torch.sum(attention_scores.unsqueeze(-1) * token_scores, dim=1)
        assert output.size() == (
            batch_size,
            self.num_classes,
        ), f"Output size {output.size()} must be equal to {(batch_size,)}"
        assert (
            attention_scores.size() == (batch_size, seq_len)
        ), f"Attention scores size {attention_scores.size()} must be equal to {(batch_size, seq_len)}"
        return output, attention_scores

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self.forward_with_scores(x)[0]

    # TODO: (keitaroskmt) Implement forward with fixed linear head nu
    def forward_with_fixed_head(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class OurViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        channels: int = 3,
    ):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.attention = OurAttention(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, image_height, image_width).
        """
        x = self.to_patch_embedding(x)
        return self.attention(x)


@hydra.main(config_path="config", config_name="main_real_setting", version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.init(project="benign_attention_real_setting")
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    seed = cfg["seed"]
    torch.manual_seed(seed)

    dataset_name = cfg["dataset"]

    if dataset_name == "cifar10":
        train_dataset, test_dataset = get_cifar10_datasets()
        image_size = 32
        num_classes = 10
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    model = OurViT(
        image_size=image_size,
        patch_size=cfg["patch_size"],
        num_classes=num_classes,
        dim=cfg["dim"],
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)

    train_sampler, test_sampler = None, None
    if cfg["use_ddp"]:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = local_rank

        setup(rank, world_size)

        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=local_rank,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = SGD(model.parameters(), lr=cfg["learning_rate"])

    # TODO: (keitaroskmt) use lr scheduler?

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    model.train()
    for epoch in range(cfg["num_epochs"]):
        if cfg["use_ddp"]:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

        sum_train_corrects, sum_train_total = evaluate(model, train_loader, device)
        sum_test_corrects, sum_test_total = evaluate(model, test_loader, device)
        if cfg["use_ddp"]:
            dist.barrier()
            dist.all_reduce(sum_train_corrects)
            dist.all_reduce(sum_train_total)
            dist.all_reduce(sum_test_corrects)
            dist.all_reduce(sum_test_total)
        train_acc = sum_train_corrects.item() / sum_train_total.item()
        test_acc = sum_test_corrects.item() / sum_test_total.item()

        if cfg["use_ddp"] and dist.get_rank() != 0:
            continue
        logger.info({"epoch": epoch, "train_acc": train_acc, "test_acc": test_acc})
        wandb.log({"train_acc": train_acc, "test_acc": test_acc})

    if cfg["use_ddp"]:
        cleanup()
    wandb.finish()


if __name__ == "__main__":
    main()
