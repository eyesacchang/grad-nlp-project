"""Run all three baselines sequentially.

Usage:
    python scripts/run_baselines.py --config configs/base_config.yaml
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from baselines.pretrained_eval import run as run_pretrained
from baselines.vanilla_infonce import run as run_vanilla
from baselines.layer_freeze import run as run_freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--skip_pretrained", action="store_true")
    parser.add_argument("--skip_vanilla", action="store_true")
    parser.add_argument("--skip_freeze", action="store_true")
    args = parser.parse_args()

    if not args.skip_pretrained:
        print("\n" + "="*60)
        print("BASELINE 1: Pretrained XLM-R (no fine-tuning)")
        print("="*60)
        run_pretrained()

    if not args.skip_vanilla:
        print("\n" + "="*60)
        print("BASELINE 2: Vanilla InfoNCE (uniform LR, no regularization)")
        print("="*60)
        run_vanilla(args.config)

    if not args.skip_freeze:
        print("\n" + "="*60)
        print("BASELINE 3: Static layer freezing (bottom 6 layers frozen)")
        print("="*60)
        run_freeze(args.config, freeze_k=6)


if __name__ == "__main__":
    main()
