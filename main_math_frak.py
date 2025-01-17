import os
import logging
import json

import numpy as np
import torch
from torch import nn, Tensor
import pandas as pd
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.nn.init import zeros_, eye_, orthogonal_
from torch.nn.parameter import Parameter
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        embed_dim: int,
        vmu_1: Tensor,
        vmu_2: Tensor,
        seq_len: int = 32,
        rho: float = 0.0,
        noise_ratio: float = 0.0,
        seed: int = 0,
    ) -> None:
        self._num_samples = num_samples

        assert vmu_1.size() == (embed_dim,), f"Signal vector vmu_1 size {vmu_1.size()} must be equal to {(embed_dim,)}"
        assert vmu_2.size() == (embed_dim,), f"Signal vector vmu_2 size {vmu_2.size()} must be equal to {(embed_dim,)}"
        assert rho > 0.0, "rho must be the positive constant"

        # Fix data generation
        np.random.seed(seed)
        data = np.random.randn(num_samples, seq_len, embed_dim)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.cat([torch.ones(num_samples // 2), -torch.ones(num_samples - num_samples // 2)])

        # First and second token is relevant token
        self.data[:, 0, :] = self.data[:, 0, :] + torch.where(self.label.view(-1, 1) == 1, vmu_1, vmu_2)
        self.data[:, 1, :] = self.data[:, 1, :] + torch.where(self.label.view(-1, 1) == 1, vmu_1, vmu_2)

        # Third and fourth token is weakly relevant token that aligns with the relevant token
        self.data[:, 2, :] = self.data[:, 2, :] + rho * torch.where(self.label.view(-1, 1) == 1, vmu_1, vmu_2)
        self.data[:, 3, :] = self.data[:, 3, :] + rho * torch.where(self.label.view(-1, 1) == 1, vmu_1, vmu_2)
        # Fifth and sixth token is weakly relevant token that aligns with the opposite direction of relevant token
        self.data[:, 4, :] = self.data[:, 4, :] + rho * torch.where(self.label.view(-1, 1) == 1, vmu_2, vmu_1)
        self.data[:, 5, :] = self.data[:, 5, :] + rho * torch.where(self.label.view(-1, 1) == 1, vmu_2, vmu_1)

        # Create noisy data
        noisy_data_size = int(np.ceil(num_samples * noise_ratio))
        assert noisy_data_size > 0 if noise_ratio > 0.0 else noisy_data_size == 0

        self._noisy_data_mask = torch.zeros(num_samples, dtype=torch.bool)
        self._noisy_data_mask[: noisy_data_size // 2] = True
        self._noisy_data_mask[num_samples // 2 : num_samples // 2 + noisy_data_size // 2] = True
        self.label = torch.where(self._noisy_data_mask, -self.label, self.label)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx], self.label[idx]

    @property
    def noisy_data_mask(self) -> Tensor:
        return self._noisy_data_mask


class Attention(nn.Module):
    def __init__(self, embed_dim: int, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.p = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))
        self.nu = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        zeros_(self.p)

    def init_linear_head(self, vmu_1: Tensor, vmu_2: Tensor, nu_norm: float) -> None:
        self.nu.data.copy_((vmu_1 - vmu_2).view(-1, 1) / torch.norm(vmu_1 - vmu_2) * nu_norm)

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 3, f"Input must have 3 dimensions, got {x.dim()}"
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding dimension {embed_dim} must match layer embedding dimension {self.embed_dim}"

        attention_logits = (x @ self.p).squeeze(-1)
        attention_scores = torch.softmax(attention_logits, dim=-1)

        token_scores = (x @ self.nu).squeeze(-1)
        output = torch.sum(attention_scores * token_scores, dim=1)
        assert output.size() == (batch_size,), f"Output size {output.size()} must be equal to {(batch_size,)}"
        return output, attention_scores

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_scores(x)[0]


def calc_accuracy_and_loss(model: nn.Module, loader: DataLoader, device: torch.device | str) -> tuple[float, float]:
    model.eval()
    correct = 0
    loss_total = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            correct += (pred * target > 0).sum().item()
            loss_total += torch.log(1.0 + torch.exp(-target * pred)).sum().item()
            total += target.size(0)
    model.train()
    return correct / total, loss_total / total


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.init(project="benign_attention")
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Same scale for the linear head as in the paper
    nu_norm = 1 / cfg["signal_norm"]

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # In this experiment, we use orthogonal basis e_1 and e_2 as signal vectors vmu_1 and vmu_2, respectively.
    vmu_1 = torch.zeros(cfg["embed_dim"])
    vmu_2 = torch.zeros(cfg["embed_dim"])
    vmu_1[0] = cfg["signal_norm"]
    vmu_2[1] = cfg["signal_norm"]

    train_dataset = CustomDataset(
        cfg["train_n"],
        cfg["embed_dim"],
        vmu_1,
        vmu_2,
        seq_len=cfg["T"],
        rho=cfg["rho"],
        noise_ratio=cfg["noise_ratio"],
        seed=cfg["seed"],
    )
    test_dataset = CustomDataset(
        cfg["test_n"],
        cfg["embed_dim"],
        vmu_1,
        vmu_2,
        seq_len=cfg["T"],
        rho=cfg["rho"],
        noise_ratio=0.0,
        seed=cfg["seed"],
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["train_n"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["test_n"])

    model = Attention(embed_dim=cfg["embed_dim"], device=device)
    model.init_linear_head(vmu_1, vmu_2, nu_norm)
    params = [model.p]
    optimizer = SGD(params, lr=cfg["learning_rate"])

    # Record attention scores and statistics
    dict_stats_time_step = {
        "time_step": [],
        "mathfrak_s_1_clean": [],
        "mathfrak_s_2_clean": [],
        "mathfrak_s_1_noisy": [],
        "mathfrak_s_2_noisy": [],
    }
    mathfrak_s_1_clean = 0.0
    mathfrak_s_1_noisy = 0.0
    mathfrak_s_2_clean = 0.0
    mathfrak_s_2_noisy = 0.0

    for time_step in range(cfg["num_steps"]):
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output, attention_scores = model.forward_with_scores(data)
            loss = torch.log(1.0 + torch.exp(-target * output)).mean()
            loss.backward()
            optimizer.step()
            assert i == 0, "Batch size must be equal to dataset size"

        # Log attention scores
        for sample_id in range(cfg["train_n"]):
            attention_score_relevant = attention_scores[sample_id, 0].item() + attention_scores[sample_id, 1].item()
            mathfrak = attention_score_relevant * (1.0 - attention_score_relevant)

            # Clean data
            if not train_dataset.noisy_data_mask[sample_id]:
                if train_dataset[sample_id][1] == 1:
                    mathfrak_s_1_clean += mathfrak
                if train_dataset[sample_id][1] == -1:
                    mathfrak_s_2_clean += mathfrak
            # Noisy data
            else:
                if train_dataset[sample_id][1] == 1:
                    mathfrak_s_1_noisy += mathfrak
                if train_dataset[sample_id][1] == -1:
                    mathfrak_s_2_noisy += mathfrak

        # relevant tokens
        wandb.log({"clean_prop": attention_scores[-1, 0].item() + attention_scores[-1, 1].item()})
        # weakly relevant tokens
        wandb.log({"noisy_prop": attention_scores[0, 4].item() + attention_scores[0, 5].item()})

        dict_stats_time_step["time_step"].append(time_step)
        dict_stats_time_step["mathfrak_s_1_clean"].append(mathfrak_s_1_clean)
        dict_stats_time_step["mathfrak_s_2_clean"].append(mathfrak_s_2_clean)
        dict_stats_time_step["mathfrak_s_1_noisy"].append(mathfrak_s_1_noisy)
        dict_stats_time_step["mathfrak_s_2_noisy"].append(mathfrak_s_2_noisy)

    # Log statistics
    train_accuracy, train_loss = calc_accuracy_and_loss(model, train_loader, device)
    test_accuracy, test_loss = calc_accuracy_and_loss(model, test_loader, device)
    logger.info(f"Train accuracy: {train_accuracy}, Train loss: {train_loss}")
    logger.info(f"Test accuracy: {test_accuracy}, Test loss: {test_loss}")

    df_stats_time_step = pd.DataFrame.from_dict(dict_stats_time_step)
    df_stats_time_step.to_csv(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "stats_time_step.csv",
        ),
        index=False,
    )

    # Output directory
    output_path = os.path.join(
        "results",
        "mathfrak",
        f"dim_{cfg['embed_dim']}",
        f"signal_{cfg['signal_norm']}",
        f"noise_{cfg['noise_ratio']}",
        f"seed_{cfg['seed']}",
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_stats_time_step.to_csv(
        os.path.join(
            output_path,
            "stats_time_step.csv",
        ),
        index=False,
    )
    with open(os.path.join(output_path, "out.json"), "w") as f:
        json.dump(
            {
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            },
            f,
        )
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    wandb.finish()


if __name__ == "__main__":
    main()
