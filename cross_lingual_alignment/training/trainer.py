import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from models.xlmr_wrapper import XLMRWrapper
from models.pretrained_snapshot import PretrainedSnapshot
from models.parameter_groups import build_parameter_groups
from data.collator import PairCollator
from losses.infonce import infonce_loss
from losses.regularization import representation_drift_loss
from alignment.scheduler import compute_lr_scales
from training.alignment_update import compute_alignment_scores, maybe_update_lr
from training.callbacks import Logger
from evaluation.geometric_eval import run_geometric_eval
from evaluation.behavioral_eval import run_behavioral_eval


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = XLMRWrapper(
            cfg["model"]["name"],
            proj_hidden=cfg["model"].get("proj_hidden", 2048),
            proj_out=cfg["model"].get("proj_out", 256),
        ).to(self.device)
        self.snapshot = PretrainedSnapshot(self.model, self.device)
        self.collator = PairCollator(cfg["model"]["name"], cfg["data"]["max_seq_len"])

        # Build initial parameter groups with uniform LR
        uniform_scales = {i: 1.0 for i in range(14)}
        param_groups = build_parameter_groups(self.model, uniform_scales, cfg["optimizer"]["base_lr"])
        self.optimizer = AdamW(
            param_groups,
            lr=cfg["optimizer"]["base_lr"],
            weight_decay=cfg["optimizer"]["weight_decay"],
            eps=cfg["optimizer"]["adam_epsilon"],
            betas=(cfg["optimizer"]["adam_beta1"], cfg["optimizer"]["adam_beta2"]),
        )

        self.scaler = GradScaler(enabled=cfg["training"]["fp16"])
        self.logger = Logger(
            use_wandb=bool(cfg["logging"].get("wandb_project")),
            wandb_project=cfg["logging"].get("wandb_project"),
            wandb_run_name=cfg["logging"].get("wandb_run_name"),
            output_dir=cfg["logging"]["output_dir"],
        )

        self.alignment_scores = {}  # most recent scores, {layer_idx: score}
        self.step = 0

    def train(self, train_data_iter, probe_pairs, eval_pairs=None):
        cfg = self.cfg
        self.model.train()

        # Compute initial alignment scores before any training
        self.alignment_scores = compute_alignment_scores(
            self.model, probe_pairs, self.collator,
            cfg["alignment"]["scorer"], self.device
        )
        self._update_lr_from_scores()

        batch_buffer = []

        for src_text, tgt_text in train_data_iter:
            batch_buffer.append((src_text, tgt_text))

            if len(batch_buffer) < cfg["data"]["train_batch_size"]:
                continue

            loss, infonce, reg = self._train_step(batch_buffer)
            batch_buffer = []
            self.step += 1

            if self.step % cfg["logging"]["log_every"] == 0:
                lr_logs = {f"lr/layer_{i}": g["lr"]
                           for i, g in enumerate(self.optimizer.param_groups)}
                self.logger.log({
                    "loss/total": loss,
                    "loss/infonce": infonce,
                    "loss/reg": reg,
                    **lr_logs,
                }, step=self.step)

            # Periodic alignment score recompute and LR update
            updated = maybe_update_lr(
                step=self.step,
                update_freq=cfg["alignment"]["update_freq"],
                model=self.model,
                optimizer=self.optimizer,
                probe_pairs=probe_pairs,
                collator=self.collator,
                scorer_type=cfg["alignment"]["scorer"],
                scheduler_type=cfg["alignment"]["scheduler_type"],
                scheduler_temperature=cfg["alignment"]["scheduler_temperature"],
                min_lr_scale=cfg["alignment"]["min_lr_scale"],
                base_lr=cfg["optimizer"]["base_lr"],
                device=self.device,
            )
            if updated:
                self.alignment_scores = updated
                self.logger.log(
                    {f"alignment/layer_{i}": v for i, v in self.alignment_scores.items()},
                    step=self.step,
                )

            if self.step % cfg["logging"]["eval_every"] == 0 and eval_pairs:
                self._run_eval(eval_pairs)

            if self.step % cfg["logging"]["save_every"] == 0:
                self._save_checkpoint()

            if self.step >= cfg["training"]["max_steps"]:
                break

        self.logger.finish()

    def _train_step(self, pairs):
        self.model.train()
        batch = self.collator(pairs)
        input_ids_src = batch["input_ids_src"].to(self.device)
        attention_mask_src = batch["attention_mask_src"].to(self.device)
        input_ids_tgt = batch["input_ids_tgt"].to(self.device)
        attention_mask_tgt = batch["attention_mask_tgt"].to(self.device)

        with autocast(enabled=self.cfg["training"]["fp16"]):
            # Final layer embeddings for contrastive loss
            z_src = self.model(input_ids_src, attention_mask_src)
            z_tgt = self.model(input_ids_tgt, attention_mask_tgt)
            i_loss = infonce_loss(z_src, z_tgt, self.cfg["loss"]["temperature"])

            # All-layer reps for regularization
            lam = self.cfg["regularization"]["lambda_reg"]
            if lam > 0:
                cur_reps = self.model.get_all_layer_reps(input_ids_src, attention_mask_src)
                frz_reps = self.snapshot.get_all_layer_reps(input_ids_src, attention_mask_src)
                # alignment_scores only passed for ablation; default is uniform weighting
                align_weights = None
                if self.cfg["regularization"]["weight_by_alignment"] and self.alignment_scores:
                    align_weights = {i - 1: v for i, v in self.alignment_scores.items()}
                r_loss = representation_drift_loss(cur_reps, frz_reps, align_weights)
                total_loss = i_loss + lam * r_loss
            else:
                r_loss = torch.tensor(0.0)
                total_loss = i_loss

        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["grad_clip_norm"])
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.item(), i_loss.item(), r_loss.item()

    def _update_lr_from_scores(self):
        full_scores = {0: 0.5, **self.alignment_scores, 13: 0.5}
        lr_scales = compute_lr_scales(
            full_scores,
            scheduler_type=self.cfg["alignment"]["scheduler_type"],
            temperature=self.cfg["alignment"]["scheduler_temperature"],
            min_scale=self.cfg["alignment"]["min_lr_scale"],
        )
        from models.parameter_groups import update_param_group_lrs
        update_param_group_lrs(self.optimizer, lr_scales, self.cfg["optimizer"]["base_lr"])

    def _run_eval(self, eval_pairs):
        geo = run_geometric_eval(self.model, self.snapshot, eval_pairs, self.collator, self.device)
        beh = run_behavioral_eval(self.model, eval_pairs, self.collator, self.device)
        self.logger.log({**geo, **beh}, step=self.step)

    def _save_checkpoint(self):
        out_dir = self.cfg["logging"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"checkpoint_step{self.step}.pt")
        torch.save({
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "alignment_scores": self.alignment_scores,
            "config": self.cfg,
        }, path)
        print(f"Saved checkpoint: {path}")
