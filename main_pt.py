import os
import logging

import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, RobertaModel
from peft import PromptTuningConfig, get_peft_model


@hydra.main(config_path="config", config_name="main_pt", version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.init(project="benign_attention_prompt_tuning")
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    seed = cfg["seed"]
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")

    peft_config = PromptTuningConfig(
        task_type="SEQ_CLS",  # TODO (keitaroskmt) compare with SEQ_CLS
        num_virtual_tokens=cfg["prompt_tuning"]["num_virtual_tokens"],  # e.g., 20
        token_dim=cfg["dim"],
        num_transformer_submodules=1,
        num_attention_heads=cfg["heads"],
        num_layers=cfg["depth"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()
