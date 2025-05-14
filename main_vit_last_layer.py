import os
import logging
import json

import torch
from torch import nn, Tensor
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)

from src.datasets.cifar import get_cifar10_hf_datasets
from src.datasets.mnist import get_mnist_snr_hf_datasets
from src.distributed_utils import setup, cleanup


def get_loss_and_logits(
    model: nn.Module,
    input,
    device: torch.device | str | int,
) -> tuple[Tensor, Tensor]:
    """
    Get the prediction tensor from the model.
    Args:
        model: Model to be used for prediction.
        input: Object from DataLoader.
        device: Device where the model is placed.
    Returns:
        tuple of loss and logits
    """
    attention_mask = (
        input["attention_mask"].to(device) if "attention_mask" in input else None
    )
    output = model(
        input["pixel_values"].to(device),
        attention_mask=attention_mask,
        labels=input["label"].to(device),
    )
    return output.loss, output.logits


def evaluate(
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
        for input in loader:
            target = input["label"].to(device)
            _, pred = get_loss_and_logits(model, input, device)
            correct += pred.argmax(dim=1).eq(target).sum()
            total += len(target)
    return correct, total


@hydra.main(config_path="config", config_name="main_vit", version_base=None)
def main(cfg: DictConfig) -> None:
    if (cfg["use_ddp"] and dist.get_rank() == 0) or not cfg["use_ddp"]:
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="benign_attention_vit_last_layer")

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

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "cifar10":
        train_dataset, test_dataset = get_cifar10_hf_datasets(
            processor=processor,
            sample_size=cfg["sample_size"],
            noise_ratio=cfg["noise_ratio"],
        )
    elif dataset_name == "mnist_snr":
        train_dataset, test_dataset = get_mnist_snr_hf_datasets(
            processor=processor,
            sample_size=cfg["sample_size"],
            noise_ratio=cfg["noise_ratio"],
            snr=cfg["dataset"]["signal_noise_ratio"],
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=cfg["dataset"]["num_classes"],
        attention_rpobs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )
    model.to(device)

    # Freeze all layers except the last layer and initialize the last layer.
    for name, param in model.named_parameters():
        if name.startswith("vit.encoder.layer.11"):
            pass
        elif name.startswith("vit.layernorm") or name.startswith("classifier"):
            pass
        else:
            param.requires_grad = False

    train_sampler, test_sampler = None, None
    if cfg["use_ddp"]:
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
        # We don't use the regularization technique, such as weight decay.
        optimizer = SGD(model.parameters(), lr=cfg["optimizer"]["learning_rate"])
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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Config: {cfg}")
    logger.info(
        f"Hydra output dir: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    losses = []
    train_accs = []
    test_accs = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        if cfg["use_ddp"]:
            assert train_sampler is not None
            train_sampler.set_epoch(epoch)
        for input in train_loader:
            optimizer.zero_grad()
            loss, _ = get_loss_and_logits(model, input, device)
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
