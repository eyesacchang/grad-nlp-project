"""Baseline 2: Vanilla InfoNCE with uniform learning rate across all layers."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from training.trainer import Trainer
from data.opus_loader import get_opus100_iterator, get_opus100_sample
from data.flores_loader import get_flores_pairs


def run(config_path="configs/base_config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override to disable adaptive behavior
    cfg["regularization"]["lambda_reg"] = 0.0
    cfg["alignment"]["update_freq"] = cfg["training"]["max_steps"] + 1  # never update
    cfg["logging"]["wandb_run_name"] = "baseline_vanilla_infonce"
    cfg["logging"]["output_dir"] = "outputs/baseline_vanilla_infonce/"

    trainer = Trainer(cfg)
    probe_pairs = get_opus100_sample(n=cfg["alignment"]["probe_batch_size"])
    eval_pairs = get_flores_pairs(src_lang="en", tgt_lang="es", split="devtest")
    train_iter = get_opus100_iterator(lang_pair=cfg["data"]["train_lang_pair"])

    trainer.train(train_iter, probe_pairs, eval_pairs)


if __name__ == "__main__":
    run()
