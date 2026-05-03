"""Diagnostic: verify that XLM-R layers vary in cross-lingual alignment quality.

This is the premise-verification experiment. Run this BEFORE full training.
If all 12 layers have similar scores, the adaptive LR signal will be noise.

Usage:
    python scripts/check_layer_alignment.py
    python scripts/check_layer_alignment.py --n_pairs 500 --scorer all
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from models.xlmr_wrapper import XLMRWrapper
from data.collator import PairCollator
from data.flores_loader import get_flores_pairs
from alignment.scorer import ProcrustesSimilarityScorer, CosineSimilarityScorer, CKAScorer


def run(n_pairs=256, scorer_types=("procrustes", "cosine", "cka"),
        model_name="xlm-roberta-base", src_lang="en", tgt_lang="es"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {model_name}...")

    model = XLMRWrapper(model_name).to(device)
    model.eval()
    collator = PairCollator(model_name)

    print(f"Loading {n_pairs} FLORES-200 {src_lang.upper()}→{tgt_lang.upper()} dev pairs...")
    all_pairs = get_flores_pairs(src_lang=src_lang, tgt_lang=tgt_lang, split="dev")
    pairs = all_pairs[:n_pairs]

    batch = collator(pairs)
    input_ids_src = batch["input_ids_src"].to(device)
    attention_mask_src = batch["attention_mask_src"].to(device)
    input_ids_tgt = batch["input_ids_tgt"].to(device)
    attention_mask_tgt = batch["attention_mask_tgt"].to(device)

    print("Extracting representations from all 12 layers...")
    with torch.no_grad():
        src_reps = model.get_all_layer_reps(input_ids_src, attention_mask_src)
        tgt_reps = model.get_all_layer_reps(input_ids_tgt, attention_mask_tgt)

    scorers = {}
    if "procrustes" in scorer_types:
        scorers["Procrustes"] = ProcrustesSimilarityScorer()
    if "cosine" in scorer_types:
        scorers["Cosine"] = CosineSimilarityScorer()
    if "cka" in scorer_types:
        scorers["CKA"] = CKAScorer()

    # Compute cross-lingual scores per layer
    results = {name: [] for name in scorers}
    intra_cos = []  # within-English cosine similarity (isotropy proxy)

    for i, (src_rep, tgt_rep) in enumerate(zip(src_reps, tgt_reps)):
        import torch.nn.functional as F
        X = src_rep.cpu()
        Y = tgt_rep.cpu()
        for name, scorer in scorers.items():
            results[name].append(scorer.score(X, Y))

        # Isotropy: mean off-diagonal cosine similarity within source language.
        # High value = anisotropic (all vectors cluster together).
        # Low value = isotropic (vectors spread across the space).
        X_norm = F.normalize(X, dim=-1)
        sim_matrix = X_norm @ X_norm.T          # (N, N)
        N = sim_matrix.size(0)
        off_diag = sim_matrix.masked_fill(torch.eye(N, dtype=torch.bool), 0.0)
        mean_intra = off_diag.sum() / (N * (N - 1))
        intra_cos.append(float(mean_intra.item()))

    results["Intra-cos"] = intra_cos

    # Print table
    all_cols = list(scorers.keys()) + ["Intra-cos"]
    header = f"{'Layer':>6}" + "".join(f"  {name:>12}" for name in all_cols)
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))
    for i in range(12):
        row = f"{i+1:>6}"
        for name in all_cols:
            row += f"  {results[name][i]:>12.4f}"
        print(row)
    print("="*len(header))

    # Print variance summary — key diagnostic
    print("\nVariance across layers (higher = more differentiation = method is useful):")
    for name in list(scorers.keys()):
        scores = results[name]
        mn, mx, rng = min(scores), max(scores), max(scores) - min(scores)
        print(f"  {name:>12}:  min={mn:.4f}  max={mx:.4f}  range={rng:.4f}  "
              f"{'✓ GOOD variance' if rng > 0.05 else '⚠ LOW variance — check method assumptions'}")

    print("\nIsotropy (Intra-cos) — within-language cosine similarity per layer:")
    print("  High = anisotropic (vectors cluster), Low = isotropic (vectors spread out)")
    for i, v in enumerate(intra_cos):
        flag = "⚠ ANISOTROPIC" if v > 0.7 else ("~ moderate" if v > 0.4 else "✓ isotropic")
        print(f"  Layer {i+1:>2}: {v:.4f}  {flag}")

    # Try to plot
    try:
        import matplotlib.pyplot as plt
        layers = list(range(1, 13))
        fig, ax1 = plt.subplots(figsize=(10, 5))

        colors = {"Procrustes": "steelblue", "Cosine": "darkorange", "CKA": "green"}
        for name in scorers:
            ax1.plot(layers, results[name], marker="o",
                     color=colors.get(name), label=f"{name} (cross-lingual)")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Cross-lingual Alignment Score")
        ax1.set_xticks(layers)
        ax1.grid(True, alpha=0.3)

        # Intra-cos on second y-axis
        ax2 = ax1.twinx()
        ax2.plot(layers, intra_cos, marker="s", linestyle="--",
                 color="crimson", label="Intra-cos (isotropy proxy)")
        ax2.set_ylabel("Within-language Cosine Similarity (↑ = more anisotropic)")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

        ax1.set_title(
            f"XLM-R Layer-wise Alignment + Isotropy "
            f"({src_lang.upper()}→{tgt_lang.upper()}, n={n_pairs})"
        )
        out_path = f"layer_alignment_{src_lang}_{tgt_lang}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=256)
    parser.add_argument("--scorer", default="all",
                        choices=["procrustes", "cosine", "cka", "all"])
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--src_lang", default="en",
                        choices=["en", "es", "fr", "de", "zh", "ar", "hi", "sw"])
    parser.add_argument("--tgt_lang", default="es",
                        choices=["en", "es", "fr", "de", "zh", "ar", "hi", "sw"])
    args = parser.parse_args()

    scorers = ("procrustes", "cosine", "cka") if args.scorer == "all" else (args.scorer,)
    run(n_pairs=args.n_pairs, scorer_types=scorers, model_name=args.model,
        src_lang=args.src_lang, tgt_lang=args.tgt_lang)
