# Notebooks

All five notebooks were developed on **Google Colab Pro with NVIDIA A100**. They share the same model class (`ColBERTEncoder`), the same `colbert_infonce_vec` loss, and the same FLORES+ evaluation utilities. They differ only in which isotropy variant is trained and which languages are evaluated.

Every notebook expects:
- HuggingFace authentication (FLORES+ is gated)
- Google Drive mounted at `/content/drive` for checkpoint storage
- A GPU runtime

## Order of execution

| # | Notebook | What it does | Runtime |
|---|---|---|---|
| 1 | [`01_main_multiseed.ipynb`](01_main_multiseed.ipynb) | Trains 6 models (3 seeds × {vanilla, IsoColBERT λ=0.5}). Evaluates on EN-ES/FR/DE/SW/AR. Saves checkpoints to `/MyDrive/iso_colbert_lam0p5/`. | ≈ 6 GPU-hours |
| 2 | [`02_low_resource_eval.ipynb`](02_low_resource_eval.ipynb) | Eval-only on the 6 checkpoints from Notebook 1. Adds YO, NE, TA. Merges into combined metrics JSON. | ≈ 30 min |
| 3 | [`03_active_token_ablation.ipynb`](03_active_token_ablation.ipynb) | Trains 3 seeds of the active-token variant. Reuses vanilla + uniform from Notebook 1. | ≈ 3 GPU-hours |
| 4 | [`04_crossattn_ablation.ipynb`](04_crossattn_ablation.ipynb) | Trains 3 seeds of the cross-attention variant. Reuses vanilla + uniform from Notebook 1. | ≈ 3 GPU-hours |
| 5 | [`05_hardneg_eval.ipynb`](05_hardneg_eval.ipynb) | LaBSE-mined hard-negative eval framework. **NOT YET RUN.** | ≈ 30–40 min if executed |

## Which results in the paper come from which notebook

| Paper artifact | Notebook |
|---|---|
| Table 1 (retrieval, 7 languages), Section 5.1 | 1 (ES/FR/DE/SW/AR) + 2 (NE/TA) |
| Per-seed paragraph in Section 5.1 | 1 (high-resource + SW + AR) + 2 (NE + TA) |
| Table 2 (geometry: intra-cos, eRank), Section 5.2 | 1 |
| Figure 2 (P@1 across 7 languages) | 1 + 2 |
| Figure 3 (low-resource pp gain bars) | 1 + 2 |
| Figure 4 (geometry, intra-cos + eRank panels) | 1 |
| Table 3 (λ sweep) | 1 (λ=0 vs λ=0.5 directly; λ=0.1 means come from an earlier multi-seed run referenced inline in cell 24 of Notebook 1) |
| Table 4 and Figure 5 (mechanism ablation) | 3 (active) + 4 (cross-attn). Uniform numbers reused from 1. |

## Conservative cleanup applied to these notebook copies

These are **cleaned copies** of the original Colab notebooks. The cleanup was deliberately minimal:
- Trailing empty cells were removed.
- A standardized header markdown cell was prepended to each notebook.
- **No code cells were modified.**
- **No outputs were cleared.**
- **No execution-count metadata was touched.**

All result tables, training logs, intermediate plots, and explanation cells appear exactly as they did in the source notebooks.

## File-naming map (cleaned → original)

| In this repo | Original filename |
|---|---|
| `01_main_multiseed.ipynb` | `iso_colbert_lam0p5_multiseed (1).ipynb` |
| `02_low_resource_eval.ipynb` | `iso_expanded.ipynb` |
| `03_active_token_ablation.ipynb` | `iso_colbert_active_token (1).ipynb` |
| `04_crossattn_ablation.ipynb` | `iso_colbert_crossattn (3).ipynb` |
| `05_hardneg_eval.ipynb` | `iso_hardneg_eval.ipynb` |
