"""Baseline 3: Static layer freezing — freeze bottom K layers, train the rest uniformly."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from training.trainer import Trainer
from data.opus_loader import get_opus100_iterator, get_opus100_sample
from data.flores_loader import get_flores_pairs
from models.xlmr_wrapper import XLMRWrapper
from models.parameter_groups import build_parameter_groups
from torch.optim import AdamW


def run(config_path="configs/base_config.yaml", freeze_k=6):
    """Train with bottom freeze_k layers frozen, top (12-freeze_k) layers at base_lr."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["regularization"]["lambda_reg"] = 0.0
    cfg["alignment"]["update_freq"] = cfg["training"]["max_steps"] + 1  # static
    cfg["logging"]["wandb_run_name"] = f"baseline_freeze_k{freeze_k}"
    cfg["logging"]["output_dir"] = f"outputs/baseline_freeze_k{freeze_k}/"

    trainer = Trainer(cfg)

    # Override: freeze bottom K encoder layers (layer indices 1..freeze_k)
    for i in range(freeze_k):
        for param in trainer.model.model.encoder.layer[i].parameters():
            param.requires_grad_(False)

    # Rebuild optimizer with only trainable params (skip frozen)
    uniform_scales = {i: 1.0 for i in range(14)}
    param_groups = build_parameter_groups(trainer.model, uniform_scales, cfg["optimizer"]["base_lr"])
    # Filter out params that don't require grad
    for g in param_groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]
    param_groups = [g for g in param_groups if g["params"]]

    trainer.optimizer = AdamW(
        param_groups,
        lr=cfg["optimizer"]["base_lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    probe_pairs = get_opus100_sample(n=cfg["alignment"]["probe_batch_size"])
    eval_pairs = get_flores_pairs(src_lang="en", tgt_lang="es", split="devtest")
    train_iter = get_opus100_iterator(lang_pair=cfg["data"]["train_lang_pair"])

    trainer.train(train_iter, probe_pairs, eval_pairs)


if __name__ == "__main__":
    run()
