import os
import logging

import wandb
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from datasets.utils import disable_progress_bar
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)
from src.datasets.mnist import (
    get_mnist_snr_hf_datasets_for_finetune,
    get_mnist_hf_datasets_for_finetune,
)
from src.datasets.cifar import get_cifar10_hf_datasets_for_finetune
from src.datasets.stl10 import get_stl10_hf_datasets_for_finetune

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="main_vit", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg["use_ddp"] or (cfg["use_ddp"] and os.environ.get("RANK", "0") == 0):
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="benign_attention_vit_finetune", config=wandb_config)

    seed = cfg["seed"]
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
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_cifar10_hf_datasets_for_finetune(
                processor=processor,
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
            )
        )
    elif dataset_name == "mnist_snr":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_mnist_snr_hf_datasets_for_finetune(
                processor=processor,
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                snr=1.0,
                noise_ratio=cfg["noise_ratio"],
            )
        )
    elif dataset_name == "mnist":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_mnist_hf_datasets_for_finetune(
                processor=processor,
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
            )
        )
    elif dataset_name == "stl10":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_stl10_hf_datasets_for_finetune(
                processor=processor,
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
            )
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

    def compute_metrics(eval_pred):
        logits = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        labels = eval_pred.label_ids
        return {"accuracy": (logits.argmax(-1) == labels).mean()}

    logger.info(f"Config: {cfg}")
    disable_progress_bar()

    ##### First training phase: pretrain the model without label noise #####
    logger.info("#############################################################")
    logger.info("First training phase: pretraining the model without label noise.")

    # Freeze all layers except the last classifier.
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=os.path.join(
            "results/vit",
            dataset_name,
            f"noise_ratio_{cfg['noise_ratio']}",
            "pretrain",
            str(seed),
        ),
        num_train_epochs=cfg["pretrain_num_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        seed=seed,
        save_safetensors=False,
        disable_tqdm=True,
        logging_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    result = trainer.evaluate()
    logger.info(result)
    trainer.save_state()

    ##### Second training phase: finetune the model with label noise #####
    logger.info("#############################################################")
    logger.info("Second training phase: finetuning the model with label noise.")
    for name, param in model.named_parameters():
        if name.startswith("vit.encoder.layer.11.attention"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=os.path.join(
            "results/vit",
            dataset_name,
            f"noise_ratio_{cfg['noise_ratio']}",
            "finetune",
            str(seed),
        ),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        seed=seed,
        save_safetensors=False,
        disable_tqdm=True,
        logging_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=finetune_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    result = trainer.evaluate()
    logger.info(result)
    trainer.save_state()

    if cfg["use_ddp"]:
        dist.destroy_process_group()

    if not cfg["use_ddp"] or (cfg["use_ddp"] and os.environ.get("RANK", "0") == 0):
        wandb.finish()


if __name__ == "__main__":
    main()
