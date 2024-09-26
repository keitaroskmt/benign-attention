import os

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


class CustomDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        embed_dim: int,
        vmu_1: Tensor,
        vmu_2: Tensor,
        rho: float = 0.0,
        noise_ratio: float = 0.0,
    ) -> None:
        self._num_samples = num_samples
        self._seq_len = seq_len
        self._embed_dim = embed_dim

        assert vmu_1.size() == (
            embed_dim,
        ), f"Signal vector vmu_1 size {vmu_1.size()} must be equal to {(embed_dim,)}"
        assert vmu_2.size() == (
            embed_dim,
        ), f"Signal vector vmu_2 size {vmu_2.size()} must be equal to {(embed_dim,)}"
        assert rho > 0.0, "rho must be the positive constant"

        self.label = torch.randint(0, 2, (num_samples,)) * 2 - 1
        self.data = torch.randn(num_samples, seq_len, embed_dim)
        # First token is relevant token
        self.data[:, 0, :] = self.data[:, 0, :] + torch.where(
            self.label.view(-1, 1) == 1, vmu_1, vmu_2
        )
        # Second token is weakly relevant token that aligns with the relevant token
        self.data[:, 1, :] = self.data[:, 1, :] + rho * torch.where(
            self.label.view(-1, 1) == 1, vmu_1, vmu_2
        )
        # Third token is weakly relevant token that aligns with the opposite direction of relevant token
        self.data[:, 2, :] = self.data[:, 2, :] + rho * torch.where(
            self.label.view(-1, 1) == 1, vmu_2, vmu_1
        )
        # Create noisy data
        noisy_data_size = int(np.ceil(num_samples * noise_ratio))
        self._noisy_data_mask = torch.zeros(num_samples, dtype=torch.bool)
        self._noisy_data_mask[:noisy_data_size] = True
        self._noisy_data_mask = self._noisy_data_mask[torch.randperm(num_samples)]
        assert noisy_data_size > 0 if noise_ratio > 0.0 else noisy_data_size == 0
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
        self.W = Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
        self.p = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))
        self.nu = Parameter(torch.empty(embed_dim, 1, **factory_kwargs))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # orthogonal_(self.W)
        eye_(self.W)
        zeros_(self.p)

    def init_linear_head(self, vmu_1: Tensor, vmu_2: Tensor, nu_norm: float) -> None:
        self.nu.data.copy_(
            (vmu_1 - vmu_2).view(-1, 1) / torch.norm(vmu_1 - vmu_2) * nu_norm
        )

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 3, f"Input must have 3 dimensions, got {x.dim()}"
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding dimension {embed_dim} must match layer embedding dimension {self.embed_dim}"

        attention_logits = (x @ self.W @ self.p).squeeze(-1)
        attention_scores = torch.softmax(attention_logits, dim=-1)

        token_scores = (x @ self.nu).squeeze(-1)
        output = torch.sum(attention_scores * token_scores, dim=1)
        assert output.size() == (
            batch_size,
        ), f"Output size {output.size()} must be equal to {(batch_size,)}"
        return output, attention_scores

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_scores(x)[0]


def calc_accuracy_and_loss(
    model: nn.Module, loader: DataLoader, device: torch.device | str
) -> tuple[float, float]:
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
        cfg["T"],
        cfg["embed_dim"],
        vmu_1,
        vmu_2,
        rho=cfg["rho"],
        noise_ratio=cfg["noise_ratio"],
    )
    test_dataset = CustomDataset(
        cfg["test_n"],
        cfg["T"],
        cfg["embed_dim"],
        vmu_1,
        vmu_2,
        rho=cfg["rho"],
        noise_ratio=0.0,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["train_n"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["test_n"])

    model = Attention(embed_dim=cfg["embed_dim"], device=device)
    model.init_linear_head(vmu_1, vmu_2, nu_norm)
    params = [model.p]
    optimizer = SGD(params, lr=cfg["learning_rate"])

    # Record attention scores and statistics
    dict_attention_scores = {
        "sample_id": [],
        "token_id": [],
        "label_flipped": [],
        "time_step": [],
        "attention_score": [],
        "cumulative_attention": [],
    }
    dict_stats_time_step = {
        "time_step": [],
        "loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "train_loss": [],
        "test_loss": [],
        "mathfrak_s_1": [],
        "mathfrak_s_2": [],
        "alignment_vmu_1": [],
        "alignment_vmu_2": [],
    }

    for time_step in range(cfg["num_steps"]):
        for _, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output, attention_scores = model.forward_with_scores(data)
            loss = torch.log(1.0 + torch.exp(-target * output)).mean()
            loss.backward()
            optimizer.step()

        # Log attention scores
        mathfrak_s_1, mathfrak_s_2 = 0.0, 0.0
        for sample_id in range(cfg["train_n"]):
            for token_id in range(cfg["T"]):
                dict_attention_scores["sample_id"].append(sample_id)
                dict_attention_scores["token_id"].append(token_id)
                dict_attention_scores["label_flipped"].append(
                    train_dataset.noisy_data_mask[sample_id].item()
                )
                dict_attention_scores["time_step"].append(time_step)
                attention_score = attention_scores[sample_id, token_id].item()
                dict_attention_scores["attention_score"].append(attention_score)
                if dict_attention_scores["cumulative_attention"]:
                    dict_attention_scores["cumulative_attention"].append(
                        dict_attention_scores["cumulative_attention"][-1]
                        + attention_score * (1.0 - attention_score)
                    )
                else:
                    dict_attention_scores["cumulative_attention"].append(
                        attention_score * (1.0 - attention_score)
                    )
            # Cauculate \mathfrak{S} using only clean data (see our paper for details)
            if not train_dataset.noisy_data_mask[sample_id]:
                if train_dataset[sample_id][1] == 1:
                    mathfrak_s_1 += dict_attention_scores["cumulative_attention"][-1]
                if train_dataset[sample_id][1] == -1:
                    mathfrak_s_2 += dict_attention_scores["cumulative_attention"][-1]

        # Log statistics
        train_accuracy, train_loss = calc_accuracy_and_loss(model, train_loader, device)
        test_accuracy, test_loss = calc_accuracy_and_loss(model, test_loader, device)

        p_norm = torch.norm(model.p).item()
        alignment_vmu_1 = model.p.squeeze()[0].item() / p_norm
        alignment_vmu_2 = model.p.squeeze()[1].item() / p_norm
        dict_stats_time_step["time_step"].append(time_step)
        dict_stats_time_step["loss"].append(loss.item())
        dict_stats_time_step["train_accuracy"].append(train_accuracy)
        dict_stats_time_step["test_accuracy"].append(test_accuracy)
        dict_stats_time_step["train_loss"].append(train_loss)
        dict_stats_time_step["test_loss"].append(test_loss)
        dict_stats_time_step["mathfrak_s_1"].append(mathfrak_s_1)
        dict_stats_time_step["mathfrak_s_2"].append(mathfrak_s_2)
        dict_stats_time_step["alignment_vmu_1"].append(alignment_vmu_1)
        dict_stats_time_step["alignment_vmu_2"].append(alignment_vmu_2)

        wandb.log(
            {
                "loss": loss.item(),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "mathfrak_s_1": mathfrak_s_1,
                "mathfrak_s_2": mathfrak_s_2,
                "alignment_vmu_1": alignment_vmu_1,
                "alignment_vmu_2": alignment_vmu_2,
            }
        )

    df_attention_scores = pd.DataFrame.from_dict(dict_attention_scores)
    df_stats_time_step = pd.DataFrame.from_dict(dict_stats_time_step)
    df_attention_scores.to_csv(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "attention_scores.csv",
        ),
        index=False,
    )
    df_stats_time_step.to_csv(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "stats_time_step.csv",
        ),
        index=False,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
