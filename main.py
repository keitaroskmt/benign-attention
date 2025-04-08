import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn.init import kaiming_uniform_, zeros_
from torch.nn.parameter import Parameter
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

import wandb


class CustomDataset(Dataset):
    """Synthetic dataset aligning with the analysis setting in the paper."""

    def __init__(  # noqa: PLR0913
        self,
        cfg: DictConfig,
        num_samples: int,
        vmu_1: Tensor,
        vmu_2: Tensor,
        noise_ratio: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = cfg["seq_len"]
        self.embed_dim = cfg["embed_dim"]
        self.rho = cfg["rho"]
        if self.rho <= 0.0:
            msg = "rho must be the positive constant"
            raise ValueError(msg)

        # Fix data generation
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((num_samples, self.seq_len, self.embed_dim))
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.cat(
            [torch.ones(num_samples // 2), -torch.ones(num_samples - num_samples // 2)],
        )

        # First token is relevant token
        self.data[:, 0, :] = self.data[:, 0, :] + torch.where(
            self.label.view(-1, 1) == 1,
            vmu_1,
            vmu_2,
        )
        # Second token is weakly relevant token that aligns with the relevant token
        self.data[:, 1, :] = self.data[:, 1, :] + self.rho * torch.where(
            self.label.view(-1, 1) == 1,
            vmu_1,
            vmu_2,
        )
        # Third token is weakly relevant token
        # that aligns with the opposite direction of relevant token
        self.data[:, 2, :] = self.data[:, 2, :] + self.rho * torch.where(
            self.label.view(-1, 1) == 1,
            vmu_2,
            vmu_1,
        )
        # Create noisy data
        noisy_data_size = int(np.ceil(num_samples * noise_ratio))
        assert noisy_data_size > 0 if noise_ratio > 0.0 else noisy_data_size == 0  # noqa: S101

        self.noisy_data_mask = torch.zeros(num_samples, dtype=torch.bool)
        self.noisy_data_mask[: noisy_data_size // 2] = True
        self.noisy_data_mask[
            num_samples // 2 : num_samples // 2 + noisy_data_size // 2
        ] = True
        self.label = torch.where(self.noisy_data_mask, -self.label, self.label)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx], self.label[idx]


class Attention(nn.Module):
    """Toy attention model aligning with the analysis setting in the paper."""

    def __init__(self, embed_dim: int, device: torch.device | None = None) -> None:
        factory_kwargs = {"device": device}
        super().__init__()

        self.embed_dim = embed_dim
        self.W = Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
        self.p = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))
        self.nu = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        kaiming_uniform_(self.W, a=5**0.5)
        zeros_(self.p)

    def init_linear_head(self, vmu_1: Tensor, vmu_2: Tensor, nu_norm: float) -> None:
        self.nu.data.copy_(
            (vmu_1 - vmu_2).view(-1, 1) / torch.norm(vmu_1 - vmu_2) * nu_norm,
        )

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() != 3:  # noqa: PLR2004
            msg = f"Input must have 3 dimensions, got {x.dim()}"
            raise ValueError(msg)
        batch_size, _, embed_dim = x.size()
        if embed_dim != self.embed_dim:
            msg = (
                f"Input embedding dimension {embed_dim} must match "
                f"layer embedding dimension {self.embed_dim}"
            )
            raise ValueError(msg)

        attention_logits = (x @ self.W @ self.p).squeeze(-1)
        attention_scores = torch.softmax(attention_logits, dim=-1)

        token_scores = (x @ self.nu).squeeze(-1)
        output = torch.sum(attention_scores * token_scores, dim=1)
        assert output.size() == (batch_size,), (  # noqa: S101
            f"Output size {output.size()} must be equal to {(batch_size,)}"
        )
        return output, attention_scores

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


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    logger.info("wandb run url: %s", run.get_url())

    # Same scale for the linear head as in the paper
    nu_norm = 1 / cfg["signal_norm"]

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

    model = Attention(embed_dim=cfg["embed_dim"], device=device)
    model.init_linear_head(vmu_1, vmu_2, nu_norm)
    params = [model.W, model.p]
    optimizer = SGD(params, lr=cfg["learning_rate"])

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

        if time_step % cfg["log_interval"] == 0:
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

    df_stats_time_step = pd.DataFrame.from_dict(dict_stats_time_step)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df_stats_time_step.to_json(output_dir / "stats_time_step.json")
    logger.info("output_dir: %s", output_dir)
    run.config["output_dir"] = str(output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
