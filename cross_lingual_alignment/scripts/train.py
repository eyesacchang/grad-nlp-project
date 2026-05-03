"""Main training entry point.

Usage:
    python scripts/train.py --config configs/base_config.yaml
    python scripts/train.py --config configs/base_config.yaml --debug --max_steps 50
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

from training.trainer import Trainer
from data.opus_loader import get_opus100_iterator, get_opus100_sample
from data.flores_loader import get_flores_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--debug", action="store_true", help="Disable W&B, set max_steps=50")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["training"]["max_steps"] = 50
        cfg["training"]["fp16"] = False
        cfg["logging"]["wandb_project"] = None
        cfg["logging"]["eval_every"] = 25
        cfg["logging"]["save_every"] = 50
        cfg["data"]["train_batch_size"] = 8
        cfg["alignment"]["probe_batch_size"] = 16
        cfg["alignment"]["update_freq"] = 25

    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    probe_pairs = get_opus100_sample(
        n=cfg["alignment"]["probe_batch_size"],
        lang_pair=cfg["data"]["train_lang_pair"],
    )
    eval_pairs = get_flores_pairs(src_lang="en", tgt_lang="es", split="devtest")
    train_iter = get_opus100_iterator(lang_pair=cfg["data"]["train_lang_pair"])

    trainer = Trainer(cfg)
    trainer.train(train_iter, probe_pairs, eval_pairs)


if __name__ == "__main__":
    main()
