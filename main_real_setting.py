import os
import logging

import torch
from torch import nn, Tensor
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.nn.init import zeros_, orthogonal_, uniform_
from torch.nn.parameter import Parameter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from einops.layers.torch import Rearrange
from transformers import get_cosine_schedule_with_warmup

from src.datasets.cifar import get_cifar10_datasets
from src.datasets.glue import get_glue_datasets
from src.datasets.agnews import get_agnews_datasets
from src.datasets.mnist import get_mnist_snr_datasets
from src.utils import add_label_noise
from src.distributed_utils import setup, cleanup


def get_pred_and_target(
    model: nn.Module,
    input,
    dataset_name: str,
    noise_ratio: float,
    num_classes: int,
    device: torch.device | str | int,
) -> tuple[Tensor, Tensor]:
    """
    Get the prediction tensor from the model.
    Args:
        model: Model to be used for prediction.
        input: Object from DataLoader.
        dataset_name: Name of the dataset.
        noise_ratio: Label noise ratio of the dataset.
        num_classes: Number of classes in the dataset.
        device: Device where the model is placed.
    Returns:
        tuple[Tensor, Tensor]: The prediction tensor and the target tensor.
    """
    if dataset_name == "cifar10" or dataset_name == "mnist_snr":
        data, target = input[0].to(device), input[1].to(device)
        target = add_label_noise(target, noise_ratio, num_classes, device)
        return model(data), target
    elif dataset_name == "sst2" or dataset_name == "agnews":
        data = input["input_ids"].to(device)
        target = input["label"].to(device)
        attention_mask = (
            input["attention_mask"].to(device) if "attention_mask" in input else None
        )
        target = add_label_noise(target, noise_ratio, num_classes, device)
        return model(data, attention_mask=attention_mask), target
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset_name: str,
    noise_ratio: float,
    num_classes: int,
    device: torch.device | str | int,
) -> tuple[Tensor, Tensor]:
    """
    Calculate the number of correct predictions and the total number of samples.
    """
    model.eval()
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    with torch.no_grad():
        for input in loader:
            pred, target = get_pred_and_target(
                model, input, dataset_name, noise_ratio, num_classes, device
            )
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

    def forward_with_scores(
        self, x: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Return the output tensor (batch_size, num_classes) and the attention scores (batch_size, seq_len).
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Attention mask tensor of shape (batch_size, seq_len).
                0 for unmasked, -inf for masked.
        """
        assert x.dim() == 3, f"Input must have 3 dimensions, got {x.dim()}"
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding dimension {embed_dim} must match layer embedding dimension {self.embed_dim}"
        assert (
            attention_mask is None or attention_mask.size() == (batch_size, seq_len)
        ), f"Attention mask size {attention_mask.size()} must be equal to {(batch_size, seq_len)}"

        # [batch_size, seq_len]
        attention_logits = (x @ self.W @ self.p).squeeze(-1)
        if attention_mask is not None:
            attention_logits = attention_logits + attention_mask
        # [batch_size, seq_len]
        attention_scores = torch.softmax(attention_logits, dim=-1)

        # [batch_size, seq_len, num_classes]
        token_scores = x @ self.nu
        output = torch.sum(attention_scores.unsqueeze(-1) * token_scores, dim=1)
        assert (
            output.size() == (batch_size, self.num_classes)
        ), f"Output size {output.size()} must be equal to {(batch_size, self.num_classes)}"
        assert (
            attention_scores.size() == (batch_size, seq_len)
        ), f"Attention scores size {attention_scores.size()} must be equal to {(batch_size, seq_len)}"
        return output, attention_scores

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self.forward_with_scores(x, attention_mask=attention_mask)[0]

    # TODO: (keitaroskmt) Implement forward with fixed linear head nu
    def forward_with_fixed_head(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ToyVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        channels: int = 3,
    ):
        super().__init__()
        height, width = size, size
        patch_height, patch_width = patch_size, patch_size

        assert (
            height % patch_height == 0 and width % patch_width == 0
        ), "Dimensions must be divisible by the patch size."

        num_patches = (height // patch_height) * (width // patch_width)
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

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        """
        assert attention_mask is None, "Attention mask is not supported for ViT."

        x = self.to_patch_embedding(x)
        return self.attention(x)


class ToyTextTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        vocab_size: int,
        dim: int,
    ):
        super().__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim
        )
        self.attention = OurAttention(dim, num_classes)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len).
            attention_mask: Attention mask tensor of shape (batch_size, seq_len).
                1 for unmasked, 0 for masked.
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.embedding_layer(x)

        if attention_mask is not None:
            if attention_mask.size() != x.size()[:2]:
                raise ValueError(
                    f"Attention mask size {attention_mask.size()} must be equal to {(x.size()[:2])}"
                )
            # 0 for unmasked, -inf for masked
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min

        return self.attention(x, attention_mask=attention_mask)


@hydra.main(config_path="config", config_name="main_real_setting", version_base=None)
def main(cfg: DictConfig) -> None:
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="benign_attention_real_setting", config=wandb_config)

    seed = cfg["seed"]
    torch.manual_seed(seed)

    if cfg["use_ddp"]:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        setup(rank, world_size)

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "cifar10":
        model = ToyVisionTransformer(
            size=cfg["dataset"]["size"],
            patch_size=cfg["patch_size"],
            num_classes=cfg["dataset"]["num_classes"],
            dim=cfg["dim"],
            channels=cfg["dataset"]["num_channels"],
        )
        train_dataset, test_dataset = get_cifar10_datasets()
    elif dataset_name == "sst2":
        model = ToyTextTransformer(
            vocab_size=30522,  # Vocab size of "bert-base-uncased" tokenizer
            num_classes=cfg["dataset"]["num_classes"],
            dim=cfg["dim"],
        )
        train_dataset, test_dataset, _ = get_glue_datasets(task_name="sst2")
    elif dataset_name == "agnews":
        model = ToyTextTransformer(
            vocab_size=30522,  # Vocab size of "bert-base-uncased" tokenizer
            num_classes=cfg["dataset"]["num_classes"],
            dim=cfg["dim"],
        )
        train_dataset, test_dataset = get_agnews_datasets()
    elif dataset_name == "mnist_snr":
        model = ToyVisionTransformer(
            size=cfg["dataset"]["size"],
            patch_size=cfg["patch_size"],
            num_classes=cfg["dataset"]["num_classes"],
            dim=cfg["dim"],
            channels=cfg["dataset"]["num_channels"],
        )
        train_dataset, test_dataset = get_mnist_snr_datasets(
            snr=cfg["dataset"]["signal_noise_ratio"]
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)

    train_sampler, test_sampler = None, None
    if cfg["use_ddp"]:
        device = local_rank
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
        batch_size=cfg["dataset"]["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["dataset"]["batch_size"],
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
    )

    if cfg["optimizer"]["name"] == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
    elif cfg["optimizer"]["name"] == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
    else:
        raise NotImplementedError(
            f"Optimizer {cfg['optimizer']['name']} is not supported."
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * cfg["warmup_epochs"],
        num_training_steps=len(train_loader) * cfg["num_epochs"],
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Config: {cfg}")

    model.train()
    for epoch in range(cfg["num_epochs"]):
        if cfg["use_ddp"]:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)
        for input in train_loader:
            optimizer.zero_grad()
            logits, target = get_pred_and_target(
                model,
                input,
                dataset_name,
                cfg["noise_ratio"],
                cfg["dataset"]["num_classes"],
                device,
            )
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        sum_train_corrects, sum_train_total = evaluate(
            model,
            train_loader,
            dataset_name,
            cfg["noise_ratio"],
            cfg["dataset"]["num_classes"],
            device,
        )
        sum_test_corrects, sum_test_total = evaluate(
            model,
            test_loader,
            dataset_name,
            cfg["noise_ratio"],
            cfg["dataset"]["num_classes"],
            device,
        )
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
