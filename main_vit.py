import logging

import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.vit import ViT
from src.datasets.cifar import get_cifar10_datasets


def calc_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device | str
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            correct += pred.argmax(dim=1).eq(target).sum().item()
    return correct / len(loader.dataset)


@hydra.main(config_path="config", config_name="main_vit", version_base=None)
def main(cfg: DictConfig) -> None:
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

    train_dataset, test_dataset = get_cifar10_datasets()
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True
    )
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
    ).to(device)

    optimizer = SGD(model.parameters(), lr=cfg["learning_rate"])

    # TODO: (keitaroskmt) use lr scheduler?

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    model.train()
    for epoch in range(cfg["num_epochs"]):
        for data, target in train_loader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

        train_acc = calc_accuracy(model, train_loader, device)
        test_acc = calc_accuracy(model, test_loader, device)
        logger.info({"epoch": epoch, "train_acc": train_acc, "test_acc": test_acc})
        wandb.log({"train_acc": train_acc, "test_acc": test_acc})
    wandb.finish()


if __name__ == "__main__":
    main()
