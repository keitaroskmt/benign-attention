import os
import logging
import json

import torch
from torch import nn, Tensor
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.nn.init import zeros_, orthogonal_, uniform_
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    ViTImageProcessor,
    ViTModel,
)
from transformers import get_cosine_schedule_with_warmup

from src.datasets.cifar import get_cifar10_hf_datasets
from src.datasets.mnist import get_mnist_snr_hf_datasets
from src.distributed_utils import setup, cleanup


def get_logits(
    cfg: DictConfig, pretrained_model: nn.Module, model: nn.Module, data: Tensor
) -> Tensor:
    with torch.no_grad():
        if cfg["feature_extractor"] == "embedding":
            feature = (
                pretrained_model(data, output_hidden_states=True, return_dict=True)
                .hidden_states[0]
                .detach()
            )
        elif cfg["feature_extractor"] == "encoder":
            feature = pretrained_model(
                data, return_dict=True
            ).last_hidden_state.detach()
        else:
            raise NotImplementedError(
                f"Feature extractor {cfg['feature_extractor']} is not supported."
            )
    return model(feature)


def evaluate(
    cfg: DictConfig,
    pretrained_model: nn.Module,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str | int,
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
            pred = get_logits(cfg, pretrained_model, model, data)
            correct += pred.argmax(dim=1).eq(target).sum()
            total += len(target)
    return correct, total


class OurAttention(nn.Module):
    """
    Attention model in the setting of our paper.
    """

    def __init__(
        self, dim: int, num_classes: int, embed_dim: int = 768, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.W = Parameter(torch.empty(embed_dim, dim, **factory_kwargs))
        self.p = Parameter(torch.empty(dim, 1, **factory_kwargs))
        self.nu = Parameter(torch.empty(embed_dim, num_classes, **factory_kwargs))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        orthogonal_(self.W)
        zeros_(self.p)
        uniform_(self.nu, -1 / self.dim**0.5, 1 / self.dim**0.5)

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


@hydra.main(config_path="config", config_name="main_vit", version_base=None)
def main(cfg: DictConfig) -> None:
    if (cfg["use_ddp"] and dist.get_rank() == 0) or not cfg["use_ddp"]:
        wandb.init(project="benign_attention_vit")
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    seed = cfg["seed"]
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if cfg["use_ddp"]:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = local_rank
        setup(rank, world_size)

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "cifar10":
        train_dataset, test_dataset = get_cifar10_hf_datasets(
            processor=processor, noise_ratio=cfg["noise_ratio"]
        )
    elif dataset_name == "mnist_snr":
        train_dataset, test_dataset = get_mnist_snr_hf_datasets(
            processor=processor,
            noise_ratio=cfg["noise_ratio"],
            snr=cfg["dataset"]["signal_noise_ratio"],
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    train_sampler, test_sampler = None, None
    if cfg["use_ddp"]:
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

    pretrained_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    pretrained_model.to(device)
    # Use only for extracting the features.
    for param in pretrained_model.parameters():
        param.requires_grad = False

    model = OurAttention(dim=cfg["dim"], num_classes=cfg["dataset"]["num_classes"])
    model = model.to(device)
    if cfg["use_ddp"]:
        pretrained_model = DDP(pretrained_model, device_ids=[local_rank])
        model = DDP(model, device_ids=[local_rank])

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
    logger.info(
        f"Hydra output dir: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    losses = []
    train_accs = []
    test_accs = []

    model.train()
    for epoch in range(cfg["num_epochs"]):
        if cfg["use_ddp"]:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            logits = get_logits(cfg, pretrained_model, model, data)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        sum_train_corrects, sum_train_total = evaluate(
            cfg, pretrained_model, model, train_loader, device
        )
        sum_test_corrects, sum_test_total = evaluate(
            cfg, pretrained_model, model, test_loader, device
        )
        if cfg["use_ddp"]:
            dist.barrier()
            dist.all_reduce(sum_train_corrects)
            dist.all_reduce(sum_train_total)
            dist.all_reduce(sum_test_corrects)
            dist.all_reduce(sum_test_total)
        train_acc = sum_train_corrects.item() / sum_train_total.item()
        test_acc = sum_test_corrects.item() / sum_test_total.item()

        if (cfg["use_ddp"] and rank == 0) or not cfg["use_ddp"]:
            logger.info(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )
            wandb.log(
                {"loss": loss.item(), "train_acc": train_acc, "test_acc": test_acc}
            )
            losses.append(loss.item())
            train_accs.append(train_acc)
            test_accs.append(test_acc)

    if (cfg["use_ddp"] and rank == 0) or not cfg["use_ddp"]:
        file_name = os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "log.json"
        )
        with open(file_name, "w") as f:
            json.dump(
                {"losses": losses, "train_accs": train_accs, "test_accs": test_accs}, f
            )

        if cfg["save_model"]:
            torch.save(
                model.state_dict(),
                os.path.join(
                    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    "last.pth",
                ),
            )

    if cfg["use_ddp"]:
        cleanup()
    wandb.finish()


if __name__ == "__main__":
    main()
