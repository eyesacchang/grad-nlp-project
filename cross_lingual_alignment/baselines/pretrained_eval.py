"""Baseline 1: Evaluate pretrained XLM-R with no fine-tuning."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.xlmr_wrapper import XLMRWrapper
from models.pretrained_snapshot import PretrainedSnapshot
from data.collator import PairCollator
from data.flores_loader import get_flores_pairs
from evaluation.geometric_eval import run_geometric_eval
from evaluation.behavioral_eval import run_behavioral_eval


def run(model_name="xlm-roberta-base"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XLMRWrapper(model_name).to(device)
    snapshot = PretrainedSnapshot(model, device)
    collator = PairCollator(model_name)

    eval_pairs = get_flores_pairs(src_lang="en", tgt_lang="es", split="devtest")
    print(f"Evaluating pretrained {model_name} on {len(eval_pairs)} FLORES-200 EN-ES pairs")

    geo = run_geometric_eval(model, snapshot, eval_pairs, collator, device)
    beh = run_behavioral_eval(model, eval_pairs, collator, device)

    print("\n--- Geometric Metrics ---")
    for k, v in sorted(geo.items()):
        print(f"  {k}: {v:.4f}")

    print("\n--- Behavioral Metrics ---")
    for k, v in sorted(beh.items()):
        print(f"  {k}: {v:.4f}")

    return {**geo, **beh}


if __name__ == "__main__":
    run()
