import os
import logging

import wandb
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)
from src.datasets.mnist import get_mnist_snr_hf_datasets
from src.datasets.cifar import get_cifar10_hf_datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="main_vit", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg["use_ddp"] or (cfg["use_ddp"] and os.environ.get("RANK", "0") == 0):
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="benign_attention_vit_last_layer", config=wandb_config)

    seed = 0
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if cfg["use_ddp"]:
        dist.init_process_group("nccl")

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
            snr=1.0,
            noise_ratio=cfg["noise_ratio"],
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=10,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )
    model = model.to(device)

    # Freeze all layers except the last layer and initialize the last layer.
    for name, param in model.named_parameters():
        if name.startswith("vit.encoder.layer.11"):
            pass
        elif name.startswith("vit.layernorm") or name.startswith("classifier"):
            pass
        else:
            param.requires_grad = False

    def compute_metrics(eval_pred):
        logits = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        labels = eval_pred.label_ids
        return {"accuracy": (logits.argmax(-1) == labels).mean()}

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
    )
    logger.info(f"Config: {cfg}")
    logger.info(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    result = trainer.evaluate()

    logger.info(result)
    if cfg["use_ddp"]:
        dist.destroy_process_group()

    if not cfg["use_ddp"] or (cfg["use_ddp"] and os.environ.get("RANK", "0") == 0):
        wandb.finish()


if __name__ == "__main__":
    main()
