# IsoColBERT: Token-Level Isotropy Regularization for Multilingual Late-Interaction Retrieval

**Authors:** Minh Tran, Isaac Chang, Ingrid Chien — Harvard University

Course project, Spring 2026.

---

## Summary

Late-interaction multilingual retrievers such as ColBERT score query-document pairs at the token level via MaxSim, placing the geometry of individual token vectors at the center of retrieval quality. We show that multilingual ColBERT trained with bitext InfoNCE produces token vectors that are anisotropic — they concentrate in a narrow cone with effective rank well below the embedding dimension, with the target-side language collapsing harder than the source-side.

To address this, we introduce **IsoColBERT**, a single regularizer term added to the InfoNCE training objective that penalizes the mean squared off-diagonal cosine similarity of valid token vectors in each batch. Zero cost at inference.

## Headline results

Across three random seeds on FLORES+, IsoColBERT (λ = 0.5):

| Language pair | Vanilla P@1 | IsoColBERT P@1 | Δ |
|---|---|---|---|
| EN–ES (trained) | 0.9973 ± 0.0010 | 0.9968 ± 0.0002 | −0.05 pp |
| EN–FR (high-resource) | 0.9983 ± 0.0006 | 0.9988 ± 0.0002 | +0.05 pp |
| EN–DE (high-resource) | 0.9957 ± 0.0006 | **0.9978 ± 0.0013** | **+0.22 pp** |
| **EN–SW (low-resource)** | 0.8344 ± 0.0230 | **0.8689 ± 0.0227** | **+3.45 pp** |
| **EN–AR (low-resource)** | 0.9423 ± 0.0204 | **0.9622 ± 0.0132** | **+1.99 pp** |
| **EN–NE (zero-shot, low-resource)** | 0.9008 ± 0.0176 | **0.9154 ± 0.0106** | **+1.46 pp** |
| **EN–TA (zero-shot, low-resource)** | 0.9066 ± 0.0133 | 0.9154 ± 0.0192 | +0.88 pp |

Token geometry shifts as designed: intra-cosine drops by 0.03 on EN and 0.11 on ES; effective rank rises by 2.44 on EN and 4.98 on ES. Mechanism ablations against active-token and cross-attention placements rule out localized and matching-level constraints as substitutes for full-manifold regularization.

## Repository layout

```
.
├── paper/                            LaTeX source of the report
│   ├── main.tex                      top-level paper file
│   ├── main.pdf                      11-page built PDF
│   ├── refs.bib                      bibliography
│   ├── sections/                     9 section .tex files
│   ├── figs/                         figure generator + 5 figures
│   ├── icml2026.{sty,bst}            ICML style files
│   └── Makefile
│
├── poster/                           poster source and final image
│   ├── poster.tex
│   ├── poster.pdf
│   ├── FINAL_POSTER.png
│   ├── make_figures.py               poster-quality figure generator
│   ├── idea.png, retrieval.png       hand-crafted figure sources
│   └── fonts/                        Titillium Web TTFs used by the poster
│
└── notebooks/                        all training and evaluation notebooks
    ├── README.md
    ├── 01_main_multiseed.ipynb       3-seed training, 5-language eval, geometry
    ├── 02_low_resource_eval.ipynb    zero-shot eval on NE / TA (and YO)
    ├── 03_active_token_ablation.ipynb
    ├── 04_crossattn_ablation.ipynb
    └── 05_hardneg_eval.ipynb         framework only (not run)
```

## Viewing the paper and poster

- Paper: open [`paper/main.pdf`](paper/main.pdf) directly.
- Poster: open [`poster/poster.pdf`](poster/poster.pdf) or the rendered PNG [`poster/FINAL_POSTER.png`](poster/FINAL_POSTER.png).

## Rebuilding the paper from source

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Style files are bundled. Requires a working `pdflatex` and `bibtex` (e.g., MacTeX or TeX Live).

## Regenerating the paper figures

```bash
cd paper/figs
python3 make_figures.py
```

Requires `matplotlib` and `numpy`. The script regenerates `fig_main_p1.pdf`, `fig_low_resource_gains.pdf`, `fig_geometric.pdf`, and `fig_ablation.pdf` from the 3-seed numbers hardcoded in the script (which match the values reported in the paper and in the notebook outputs).

## Reproducing the experiments

Every notebook in `notebooks/` was run on Google Colab Pro with an NVIDIA A100. Each training run is about one GPU-hour; eval-only notebooks are minutes.

The notebooks expect a Google Drive folder layout for checkpoint storage:

```
/MyDrive/iso_colbert_lam0p5/                       weights and metrics from Notebook 1
/MyDrive/iso_colbert_active_token/                 weights from Notebook 3
/MyDrive/iso_colbert_crossattn/                    weights from Notebook 4
```

Notebooks 2, 3, 4 reuse the vanilla and uniform-IsoColBERT checkpoints produced by Notebook 1; they copy them in at the top of the notebook.

To reproduce from scratch you would, in order:
1. Run **`01_main_multiseed.ipynb`** — produces vanilla and IsoColBERT λ=0.5 checkpoints for 3 seeds.
2. Run **`02_low_resource_eval.ipynb`** — eval-only, adds YO/NE/TA numbers.
3. Run **`03_active_token_ablation.ipynb`** — trains the 3-seed active-token variant.
4. Run **`04_crossattn_ablation.ipynb`** — trains the 3-seed cross-attention variant.
5. Optionally run **`05_hardneg_eval.ipynb`** — eval-only against LaBSE-mined hard negatives. Not run for the submitted paper; provided as a working framework for future work.

All five notebooks share the same `ColBERTEncoder` class, the same `colbert_infonce_vec` loss, and the same FLORES+ evaluation utilities. They differ only in which isotropy variant is being trained and which languages are being evaluated.

See [`notebooks/README.md`](notebooks/README.md) for a per-notebook breakdown.

## Method in one paragraph

We add a single regularizer term to the standard symmetric in-batch InfoNCE loss used to train multilingual ColBERT:

```
L = L_InfoNCE + λ · ½ · (R(X_Q) + R(X_D))
R(X) = (1/(N(N−1))) · Σ_{i ≠ j} ⟨xᵢ, xⱼ⟩²
```

`X_Q` and `X_D` are the pooled, L2-normalized valid token vectors of all queries and documents in the current mini-batch. The penalty pushes token vectors toward orthogonality across the full embedding manifold. We compare this *uniform* placement against two alternatives (active-token, cross-attention) in the ablations.

The regularizer adds one matrix multiplication per training step (≤ 17M FLOPs per side at N ≤ 256, k = 128). At inference the regularizer is removed entirely — the deployment graph is identical to a vanilla ColBERT encoder.

## Limitations

We trained on a single language pair (EN-ES) and tested transfer to six others. We did not test multi-pair training, larger encoders beyond XLM-RoBERTa-base, or retrieval-specialized backbones (mE5). The benchmark is FLORES+ only; MIRACL and mMARCO remain to be evaluated. Three seeds is the practical minimum we report; per-seed numbers are in the notebooks. The λ grid is coarse: {0, 0.1, 0.5}.

## Citing this work

If you use IsoColBERT or build on this code:

```bibtex
@misc{tran2026isocolbert,
  title  = {IsoColBERT: Token-Level Isotropy Regularization for Multilingual Late-Interaction Retrieval},
  author = {Tran, Minh and Chang, Isaac and Chien, Ingrid},
  year   = {2026},
  note   = {Course project, Harvard University}
}
```
