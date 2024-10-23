import os
import logging

import torch
from torch import nn, Tensor
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(config_path="config", config_name="main_vit", version_base=None)
def main(cfg: DictConfig) -> None:
    if (cfg["use_ddp"] and dist.get_rank() == 0) or not cfg["use_ddp"]:
        wandb.init(project="benign_attention_vit")
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    seed = cfg["seed"]
    torch.manual_seed(seed)

    train_dataset, test_dataset = get_cifar10_datasets()
    image_size = 32
    num_classes = 10

    # TODO: (keitaroskmt) Add label noise

    model = ViT(
        image_size=image_size,
        patch_size=cfg["patch_size"],
        num_classes=num_classes,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        mlp_dim=cfg["mlp_dim"],
        dropout=cfg["dropout"],
        emb_dropout=cfg["emb_dropout"],
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
