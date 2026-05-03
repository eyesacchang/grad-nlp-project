"""Standalone evaluation on a saved checkpoint.

Usage:
    python scripts/eval.py --checkpoint outputs/checkpoint_step5000.pt --split devtest
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from models.xlmr_wrapper import XLMRWrapper
from models.pretrained_snapshot import PretrainedSnapshot
from data.collator import PairCollator
from data.flores_loader import get_flores_pairs
from evaluation.geometric_eval import run_geometric_eval
from evaluation.behavioral_eval import run_behavioral_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--src_lang", default="en")
    parser.add_argument("--tgt_lang", default="es")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = XLMRWrapper(cfg["model"]["name"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    snapshot = PretrainedSnapshot(model, device)
    collator = PairCollator(cfg["model"]["name"], cfg["data"]["max_seq_len"])

    eval_pairs = get_flores_pairs(src_lang=args.src_lang, tgt_lang=args.tgt_lang, split=args.split)
    print(f"Evaluating {args.checkpoint} on FLORES-200 {args.src_lang}-{args.tgt_lang} ({args.split})")
    print(f"  N pairs: {len(eval_pairs)}")

    geo = run_geometric_eval(model, snapshot, eval_pairs, collator, device)
    beh = run_behavioral_eval(model, eval_pairs, collator, device)

    print("\n--- Geometric ---")
    for k, v in sorted(geo.items()):
        print(f"  {k}: {v:.4f}")

    print("\n--- Behavioral ---")
    for k, v in sorted(beh.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
