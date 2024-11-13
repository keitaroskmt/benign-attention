import os
import logging

import wandb
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from src.datasets.glue import get_glue_datasets_for_finetune
from src.datasets.agnews import get_agnews_datasets_for_finetune

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="main_bert", version_base=None)
def main(cfg: DictConfig) -> None:
    if not cfg["use_ddp"] or (cfg["use_ddp"] and os.environ.get("RANK", "0") == 0):
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project="benign_attention_bert_finetune", config=wandb_config)

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

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "sst2":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_glue_datasets_for_finetune(
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
                task_name="sst2",
            )
        )
    elif dataset_name == "agnews":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_agnews_datasets_for_finetune(
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
            )
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=cfg["dataset"]["num_classes"],
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
        output_dir="./results",
        num_train_epochs=cfg["pretrain_num_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
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

    # Freeze all layers except the last attention layer.
    for name, param in model.named_parameters():
        # TODO: whether containining `name.startswith("bert.pooler")` or not.
        if name.startswith("bert.encoder.layer.11.attention.attention"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
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
