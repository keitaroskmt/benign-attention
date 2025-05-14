import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import tensor
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

import wandb
from datasets.utils import disable_progress_bar
from src.datasets.agnews import get_agnews_datasets_for_finetune
from src.datasets.glue import get_glue_datasets_for_finetune
from src.datasets.trec import get_trec_datasets_for_finetune
from src.hf_utils import AttentionScoreCallback


@hydra.main(config_path="config", config_name="main_bert", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: C901, PLR0912, PLR0915
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

    seed = cfg["seed"]
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "sst2":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_glue_datasets_for_finetune(
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
                task_name="sst2",
                seed=seed,
            )
        )
    elif dataset_name == "agnews":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_agnews_datasets_for_finetune(
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
                seed=seed,
            )
        )
    elif dataset_name == "trec":
        pretrain_dataset, finetune_dataset, test_dataset = (
            get_trec_datasets_for_finetune(
                pretrain_sample_size=cfg["pretrain_sample_size"],
                sample_size=cfg["sample_size"],
                noise_ratio=cfg["noise_ratio"],
                seed=seed,
            )
        )
    else:
        msg = f"Dataset {dataset_name} is not supported."
        raise NotImplementedError(msg)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=cfg["dataset"]["num_classes"],
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )
    model = model.to(device)

    def compute_metrics(eval_pred) -> dict[str, tensor]:  # noqa: ANN001
        logits = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        labels = eval_pred.label_ids
        return {"accuracy": (logits.argmax(-1) == labels).mean()}

    logger.info("Config: %(config)s", extra={"config": cfg})
    disable_progress_bar()

    ##### First training phase: pretrain the model without label noise #####
    logger.info("#############################################################")
    logger.info("First training phase: pretraining the model without label noise.")
    model_save_dir = Path("results/bert") / dataset_name / "pretrain" / f"seed_{seed}"

    if not model_save_dir.exists():
        # Freeze all layers except the last classifier.
        for name, param in model.named_parameters():
            if name.startswith(("bert.pooler", "classifier")):
                param.requires_grad = True
            else:
                param.requires_grad = False

        training_args = TrainingArguments(
            output_dir=Path("results/bert")
            / dataset_name
            / "pretrain"
            / f"seed_{seed}",
            num_train_epochs=cfg["pretrain_num_epochs"],
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            seed=seed,
            save_strategy="no",
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
        trainer.save_model(model_save_dir)
    else:
        logger.info(
            "Pretrained model already exists in %(model_save_dir)s. Skip the pretraining phase.",
            extra={"model_save_dir": model_save_dir},
        )
        model = BertForSequenceClassification.from_pretrained(
            model_save_dir,
            num_labels=cfg["dataset"]["num_classes"],
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        model = model.to(device)

    ##### Second training phase: finetune the model with label noise #####
    logger.info("#############################################################")
    logger.info("Second training phase: finetuning the model with label noise.")

    # Freeze all layers except the last attention layer.
    for name, param in model.named_parameters():
        if name.startswith(
            (
                "bert.encoder.layer.11.attention.self.key",
                "bert.encoder.layer.11.attention.self.query",
            ),
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if cfg["revert_weights"]:
        for name, module in model.named_modules():
            if name.startswith(
                (
                    "bert.encoder.layer.11.attention.self.key",
                    "bert.encoder.layer.11.attention.self.query",
                ),
            ):
                if not hasattr(module, "reset_parameters"):
                    msg = f"Module {name} does not have a 'reset_parameters' method."
                    raise AttributeError(msg)
                module.reset_parameters()

    training_args = TrainingArguments(
        output_dir=Path("results/bert")
        / dataset_name
        / "finetune"
        / f"noise_ratio_{cfg['noise_ratio']}"
        / f"sample_size_{cfg['sample_size']}"
        / f"revert_weights_{cfg['revert_weights']}"
        / f"seed_{seed}",
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        seed=seed,
        save_strategy="no",
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
        callbacks=[AttentionScoreCallback()],
    )
    trainer.train()
    result = trainer.evaluate()
    logger.info(result)
    wandb.log({f"eval_test/{k}": v for k, v in result.items()})

    train_result = trainer.evaluate(eval_dataset=finetune_dataset)
    logger.info("train dataset result")
    logger.info(train_result)
    wandb.log({f"eval_train/{k}": v for k, v in train_result.items()})

    trainer.save_state()
    run.config["model_save_dir"] = str(model_save_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
