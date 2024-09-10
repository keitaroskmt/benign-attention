import numpy as np
import torch
from torch import nn, Tensor
import pandas as pd
import wandb

from torch.nn.init import zeros_, orthogonal_, eye_
from torch.nn.parameter import Parameter
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(
        self, num_samples: int, seq_len: int, embed_dim: int, vmu_1: Tensor, vmu_2: Tensor, noise_ratio: float = 0.0
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        assert vmu_1.size() == (embed_dim,), f"Signal vector vmu_1 size {vmu_1.size()} must be equal to {(embed_dim,)}"
        assert vmu_2.size() == (embed_dim,), f"Signal vector vmu_2 size {vmu_2.size()} must be equal to {(embed_dim,)}"

        self.label = torch.randint(0, 2, (num_samples,)) * 2 - 1
        self.data = torch.randn(num_samples, seq_len, embed_dim)
        # Add signal vectors to the first token of each sample
        self.data[:, 0, :] = self.data[:, 0, :] + torch.where(self.label.view(-1, 1) == 1, vmu_1, vmu_2)
        # Create noisy data
        noisy_data_size = int(np.ceil(num_samples * noise_ratio))
        self.noisy_data_mask = torch.zeros(num_samples, dtype=torch.bool)
        self.noisy_data_mask[:noisy_data_size] = True
        self.noisy_data_mask = self.noisy_data_mask[torch.randperm(num_samples)]
        assert noisy_data_size > 0 if noise_ratio > 0.0 else noisy_data_size == 0
        self.label = torch.where(self.noisy_data_mask, -self.label, self.label)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx], self.label[idx]


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
        self.nu.data.copy_((vmu_1 - vmu_2).view(-1, 1) / torch.norm(vmu_1 - vmu_2) * nu_norm)

    def forward_with_scores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 3, f"Input must have 3 dimensions, got {x.dim()}"
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding dimension {embed_dim} must match layer embedding dimension {self.embed_dim}"

        attention_logits = (x @ self.W @ self.p).squeeze()
        attention_scores = torch.softmax(attention_logits, dim=-1)

        token_scores = (x @ self.nu).squeeze()
        output = torch.sum(attention_scores * token_scores, dim=1).squeeze()
        assert output.size() == (batch_size,), f"Output size {output.size()} must be equal to {(batch_size,)}"
        return output, attention_scores

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_scores(x)[0]


def calc_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            correct += (pred * target > 0).sum().item()
            total += target.size(0)
    model.train()
    return correct / total


if __name__ == "__main__":
    # Setting parameters
    config = dict(
        delta=0.01,
        delta_r=0.1,
        n=8,
        T=8,
        embed_dim=2000,
        learning_rate=1e-4,
        signal_norm=20.0,
        num_steps=10000,
        noise_ratio=0.1,
        num_test_samples=1000,
    )
    nu_norm = 1 / config["signal_norm"]

    wandb.init(project="benign_overfitting_attention", config=config)

    # ndb device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # In this experiment, we use orthogonal basis e_1 and e_2 as signal vectors vmu_1 and vmu_2, respectively.
    vmu_1 = torch.zeros(config["embed_dim"])
    vmu_2 = torch.zeros(config["embed_dim"])
    vmu_1[0] = config["signal_norm"]
    vmu_2[1] = config["signal_norm"]

    train_dataset = CustomDataset(
        config["n"], config["T"], config["embed_dim"], vmu_1, vmu_2, noise_ratio=config["noise_ratio"]
    )
    test_dataset = CustomDataset(
        config["num_test_samples"], config["T"], config["embed_dim"], vmu_1, vmu_2, noise_ratio=0.0
    )
    train_loader = DataLoader(train_dataset, batch_size=config["n"])
    test_loader = DataLoader(test_dataset, batch_size=config["n"])

    model = Attention(embed_dim=config["embed_dim"], device=device)
    model.init_linear_head(vmu_1, vmu_2, nu_norm)
    params = [model.p]
    optimizer = SGD(params, lr=config["learning_rate"])
    dict_attention_scores = {
        "sample_id": [],
        "token_id": [],
        "label_flipped": [],
        "time_step": [],
        "attention_score": [],
    }

    for time_step in range(config["num_steps"]):
        for _, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output, attention_scores = model.forward_with_scores(data)
            loss = torch.log(1.0 + torch.exp(-target * output)).mean()
            loss.backward()
            optimizer.step()
        wandb.log(
            {
                "loss": loss.item(),
                "train_accuracy": calc_accuracy(model, train_loader, device),
                "test_accuracy": calc_accuracy(model, test_loader, device),
            }
        )

        for sample_id in range(config["n"]):
            for token_id in range(config["T"]):
                dict_attention_scores["sample_id"].append(sample_id)
                dict_attention_scores["token_id"].append(token_id)
                dict_attention_scores["label_flipped"].append(train_dataset.noisy_data_mask[sample_id].item())
                dict_attention_scores["time_step"].append(time_step)
                dict_attention_scores["attention_score"].append(attention_scores[sample_id, token_id].item())

    df_attention_scores = pd.DataFrame.from_dict(dict_attention_scores)
    df_attention_scores.to_csv("attention_scores.csv", index=False)
