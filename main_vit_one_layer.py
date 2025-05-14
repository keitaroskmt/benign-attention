import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from einops import repeat
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

import wandb
from main import CustomDataset
from src.models.vit import FeedForward


class SingleHeadAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.scale = dim ** (-0.5)
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        return out, attn


class OneLayerViT(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        use_skip_connection: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        super().__init__()
        self.embed_dim = dim
        self.use_skip_connection = use_skip_connection
        self.attn = SingleHeadAttention(dim)
        self.ff = FeedForward(dim=dim, hidden_dim=mlp_dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # Corresponds to p

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() != 3:  # noqa: PLR2004
            msg = f"Input must have 3 dimensions, got {x.dim()}"
            raise ValueError(msg)
        batch_size, _, embed_dim = x.size()
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        out, attention_scores = self.attn.forward_with_scores(x)
        # Fetch softmax probabilities for the query at the position of class token
        attention_scores = attention_scores[:, 0, 1:]
        if self.use_skip_connection:
            x = out + x
            x = self.ff(x) + x
        else:
            x = self.ff(out)
        x = self.norm(x)
        x = self.head(x[:, 0]).squeeze(1)
        return x, attention_scores

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_scores(x)[0]


def calc_accuracy_and_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    loss_total = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            pred = model(data)
            correct += (pred * target > 0).sum().item()
            loss_total += torch.log(1.0 + torch.exp(-target * pred)).sum().item()
            total += target.size(0)
    model.train()
    return correct / total, loss_total / total


@hydra.main(config_path="config", config_name="main_vit_one_layer", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type=cfg.wandb.job_type,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    logger.info("wandb run url: %s", run.get_url())

    # In this experiment, we use orthogonal basis e_1 and e_2
    # as signal vectors vmu_1 and vmu_2, respectively.
    vmu_1 = torch.zeros(cfg["embed_dim"])
    vmu_2 = torch.zeros(cfg["embed_dim"])
    vmu_1[0] = cfg["signal_norm"]
    vmu_2[1] = cfg["signal_norm"]

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_dataset = CustomDataset(
        cfg=cfg,
        num_samples=cfg["train_n"],
        vmu_1=vmu_1,
        vmu_2=vmu_2,
        noise_ratio=cfg["noise_ratio"],
        seed=cfg["seed"],
    )
    test_dataset = CustomDataset(
        cfg=cfg,
        num_samples=cfg["test_n"],
        vmu_1=vmu_1,
        vmu_2=vmu_2,
        noise_ratio=0.0,
        seed=cfg["seed"],
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["train_n"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["test_n"])

    model = OneLayerViT(
        dim=cfg["embed_dim"],
        mlp_dim=cfg["mlp_dim"],
        use_skip_connection=cfg["use_skip_connection"],
    ).to(device)
    optimizer = SGD(model.parameters(), lr=cfg["learning_rate"])

    # Record attention scores and statistics
    dict_stats_time_step = {
        "time_step": [],
        "loss": [],
        "attention_score": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": [],
    }

    for time_step in range(cfg["num_steps"]):
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            output, attention_scores = model.forward_with_scores(data)
            loss = torch.log(1.0 + torch.exp(-target * output)).mean()
            loss.backward()
            optimizer.step()
            if i > 0:
                msg = "Batch size must be equal to dataset size"
                raise ValueError(msg)

        dict_stats_time_step["time_step"].append(time_step)
        dict_stats_time_step["loss"].append(loss.item())
        dict_stats_time_step["attention_score"].append(attention_scores.tolist())

        if time_step % cfg["log_interval"] == cfg["log_interval"] - 1:
            # Log statistics
            train_accuracy, train_loss = calc_accuracy_and_loss(
                model,
                train_loader,
                device,
            )
            test_accuracy, test_loss = calc_accuracy_and_loss(
                model,
                test_loader,
                device,
            )

            dict_stats_time_step["train_accuracy"].append(train_accuracy)
            dict_stats_time_step["test_accuracy"].append(test_accuracy)
            dict_stats_time_step["train_loss"].append(train_loss)
            dict_stats_time_step["test_loss"].append(test_loss)
            wandb.log(
                {
                    "loss": loss.item(),
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
            )
            logger.info(
                "Time step %d: loss %.4f, train accuracy %.4f, test accuracy %.4f",
                time_step,
                loss.item(),
                train_accuracy,
                test_accuracy,
            )
        else:
            dict_stats_time_step["train_accuracy"].append(np.nan)
            dict_stats_time_step["test_accuracy"].append(np.nan)
            dict_stats_time_step["train_loss"].append(np.nan)
            dict_stats_time_step["test_loss"].append(np.nan)

    if not cfg["log_attention_score"]:
        dict_stats_time_step["attention_score"] = [np.nan] * len(
            dict_stats_time_step["time_step"],
        )
    df_stats_time_step = pd.DataFrame.from_dict(dict_stats_time_step)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df_stats_time_step.to_json(output_dir / "stats_time_step.json")
    logger.info("output_dir: %s", output_dir)
    run.config["output_dir"] = str(output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
