import torch
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

import wandb


class TrainerWithAttentionScores(Trainer):
    def evaluate(self, eval_dataset=None, **kwargs) -> dict[str, float]:
        # Run the default evaluation loop first to get metrics like accuracy
        output = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        # Now re-run on eval dataset just to gather attention stats
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        all_max_cls_attn = []

        for batch in dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}  # noqa: PLW2901
            with torch.no_grad():
                outputs = self.model(**batch, output_attentions=True)
                attentions = outputs.attentions  # tuple of (layers, B, H, L, L)
                final_layer_attn = attentions[-1]  # (B, H, L, L)
                cls_attn = final_layer_attn[:, :, 0, :]  # (B, H, L)
                max_scores = cls_attn.max(dim=-1).values  # (B, H)
                all_max_cls_attn.append(max_scores)

        # Concatenate and average
        all_max_cls_attn = torch.cat(all_max_cls_attn, dim=0)  # (N, H)
        mean_max_cls_attn = all_max_cls_attn.mean().item()

        # Log to W&B and also return in `metrics`
        wandb.log({"mean_max_cls_attention": mean_max_cls_attn}, commit=False)
        output["mean_max_cls_attention"] = mean_max_cls_attn
        return output


class AttentionScoreCallback(TrainerCallback):
    def on_epoch_end(
        self,
        args,
        state,
        control,
        model=None,
        train_dataloader=None,
        **kwargs,
    ):
        if train_dataloader is None or model is None:
            return

        model.eval()
        all_max_cls_attn = []

        for batch in train_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}  # noqa: PLW2901
            with torch.no_grad():
                outputs = model(**batch, output_attentions=True)
                attentions = outputs.attentions  # tuple of (layers, B, H, L, L)
                final_layer_attn = attentions[-1]  # (B, H, L, L)
                cls_attn = final_layer_attn[:, :, 0, :]  # (B, H, L)
                max_scores = cls_attn.max(dim=-1).values  # (B, H)
                all_max_cls_attn.append(max_scores)

        # Concatenate and average
        all_max_cls_attn = torch.cat(all_max_cls_attn, dim=0)  # (N, H)
        mean_max_cls_attn = all_max_cls_attn.mean().item()
        wandb.log({"mean_max_cls_attention": mean_max_cls_attn}, commit=False)
