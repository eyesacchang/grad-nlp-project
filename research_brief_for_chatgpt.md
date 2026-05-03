# Research Context Document: Adaptive Cross-Lingual Fine-Tuning via Layer-Wise Alignment Geometry

---

## Section 1: Project Summary

This project introduces an adaptive fine-tuning framework for multilingual language models, targeting the problem of cross-lingual alignment in transformer representations. The base model is **XLM-RoBERTa (XLM-R base, 12 layers)**, a state-of-the-art multilingual pretrained model that exhibits partial cross-lingual alignment — semantically equivalent sentences in different languages tend to cluster in nearby embedding regions, but this alignment is non-uniform across layers.

The core insight is that different transformer layers vary significantly in their cross-lingual alignment quality. Prior fine-tuning methods ignore this structure, applying uniform learning rates across all layers. This project proposes a data-driven alternative: compute an alignment score for each layer using geometric metrics (Procrustes distance, cosine similarity, or Centered Kernel Alignment), then **scale each layer's learning rate inversely to its alignment score**. Poorly aligned layers receive larger updates; well-aligned layers receive smaller updates plus stronger regularization to prevent representational drift.

Training uses **InfoNCE contrastive loss with in-batch negatives** on OPUS-100 English–Spanish parallel bitext. Evaluation combines geometric measures (layer-wise Procrustes distance, CKA relative to pretrained representations) and behavioral measures (cross-lingual sentence retrieval P@1/5/10 on FLORES-200), tested on both the training language pair (EN–ES) and unseen bystander languages.

---

## Section 2: Key Technical Concepts

- **Cross-lingual alignment**: The degree to which representations of semantically equivalent content in two languages occupy similar regions of embedding space. Measured as similarity between mean-pooled layer representations of translation pairs.

- **Procrustes analysis / Procrustes distance**: An orthogonal matrix alignment technique. Given source representation matrix X and target Y, finds the optimal rotation W minimizing ||X − YW||_F. Procrustes distance ∈ [0, 2]; alignment score = 1 − distance/2 ∈ [0, 1].

- **Centered Kernel Alignment (CKA)**: A similarity index for comparing neural network representations that is invariant to orthogonal transformation and isotropic scaling. Linear CKA = ||Y^T X||_F² / (||X^T X||_F · ||Y^T Y||_F). Returns value in [0, 1].

- **Discriminative fine-tuning**: Using different learning rates for different layers of a pretrained model (Howard & Ruder, 2018 / ULMFiT). Typically depth-based (lower LR for early layers). This project replaces depth-heuristics with geometry-driven LR scaling.

- **InfoNCE loss**: A contrastive objective (van den Oord et al.) maximizing mutual information. For a batch of B translation pairs, computes a B×B similarity matrix and treats the diagonal as positives. Loss = symmetric cross-entropy over the similarity matrix. Temperature τ controls sharpness.

- **Representational drift**: Change in a pretrained model's internal representations caused by fine-tuning. Catastrophic drift in well-aligned layers degrades cross-lingual transfer to bystander languages. Controlled here via an L2 regularization term weighted by per-layer alignment scores.

- **Layer-wise relevance in transformers**: The empirical finding that different transformer layers encode different levels of linguistic abstraction. For multilingual models, cross-lingual alignment tends to be strongest in middle layers and weaker at early/late layers — but this varies by model and language pair.

---

## Section 3: Known Papers (Do Not Re-Suggest These)

| Citation | Year | Venue | Role in This Project |
|---|---|---|---|
| Conneau et al. — XLM-RoBERTa | 2020 | EMNLP | Base model used for all experiments |
| Howard & Ruder — ULMFiT | 2018 | ACL | Discriminative fine-tuning (depth-based LR scaling) — inspiration for our geometry-based alternative |
| Bakos et al. — AlignFreeze | 2025 | ? | Closest prior work: selectively freezes well-aligned layers (binary freeze vs. our continuous scaling) |
| Kornblith et al. — CKA | 2019 | ICML | Evaluation metric for representational similarity; also candidate training signal |
| Gao et al. — SimCSE | 2021 | EMNLP | Contrastive sentence embeddings; InfoNCE-based objective |
| Costa-jussà et al. — FLORES-200 / NLLB | 2022 | TACL | Evaluation dataset (multilingual parallel benchmark) |
| Zhang et al. — OPUS-100 | 2020 | LREC | Training dataset (parallel bitext, EN–ES subset) |
| van den Oord et al. — CPC / InfoNCE | 2018 | ArXiv | Origin of InfoNCE contrastive objective |

---

## Section 4: Search Queries for ChatGPT to Execute

Please run each of the following queries as independent literature searches, covering NeurIPS, ICLR, ICML, ACL, EMNLP, NAACL, and ArXiv cs.CL/cs.LG from **2020 to present**. For each paper found, use the output format in Section 7.

**Query 1 — Layer-wise LR × multilingual:**
Search NeurIPS 2022–2024, ICLR 2022–2025, and ICML 2022–2024 for papers that combine layer-wise or parameter-wise learning rate adaptation with multilingual or cross-lingual training objectives. Include papers on selective layer updating, gradient surgery for multilingual models, or per-parameter importance weighting in multilingual settings.

**Query 2 — Cross-lingual alignment geometry:**
Find papers from ACL, EMNLP, NAACL 2020–2024 that measure or improve cross-lingual alignment geometry using Procrustes analysis, CKA, representational similarity analysis (RSA), or related metrics. Exclude papers that only use these metrics for post-hoc analysis without any training intervention.

**Query 3 — Representational drift + regularization in multilingual FT:**
Find papers studying representational drift or catastrophic forgetting during fine-tuning of multilingual pretrained models (XLM-R, mBERT, LaBSE, etc.), particularly those proposing regularization methods to preserve pretrained structure. Include elastic weight consolidation (EWC) applied to multilingual models.

**Query 4 — Adaptive LR schedules driven by representation properties:**
Find papers proposing adaptive or dynamic learning rate schedules for fine-tuning transformers where the schedule is driven by some property of the learned representations (e.g., representation similarity, alignment quality, layer importance), as opposed to gradient magnitude, loss value, or depth heuristics alone.

**Query 5 — InfoNCE / NT-Xent for multilingual representation learning:**
Search for papers using InfoNCE, NT-Xent, or symmetric cross-entropy contrastive loss for multilingual representation learning. Specifically look for papers that analyze which transformer layers are most affected by contrastive fine-tuning, or that study temperature sensitivity in multilingual contrastive settings.

**Query 6 — Layer-by-layer geometry of multilingual models:**
Find papers that analyze the geometry of multilingual models (XLM-R, mBERT, LaBSE, mT5) layer by layer, reporting alignment quality, isotropy, or representational similarity metrics per layer. Especially interested in papers that identify which specific layers are most/least cross-lingually aligned prior to any fine-tuning.

**Query 7 — Discriminative fine-tuning analysis:**
Find papers from 2019–2025 comparing discriminative fine-tuning strategies in NLP (assigning different learning rates to different layers), especially those with empirical or theoretical analysis of *why* layer-differentiated learning rates help, or papers that propose alternatives to depth-based heuristics for choosing per-layer LRs.

**Query 8 — Layer-adaptive weight decay / representation-aware regularization:**
Search ArXiv cs.CL and cs.LG from 2022–2025 for papers proposing layer-adaptive or representation-aware weight decay, L2 regularization, or other forms of parameter protection during fine-tuning that vary the regularization strength across layers based on learned representations or task-relevance scores.

---

## Section 5: Literature Gaps to Investigate

These are the specific novelty claims of this project. Please search specifically to determine whether any prior work already addresses these:

1. **Geometry-driven layer LR scaling**: Has any paper used an explicit geometric alignment score (Procrustes, cosine, CKA — not gradient-based Fisher information or Hessian approximations) to dynamically set per-layer learning rates during multilingual fine-tuning?

2. **Procrustes as a training signal**: Has Procrustes distance been used as an *online training signal* that drives optimization decisions (e.g., LR scaling, gradient weighting), as opposed to only a post-hoc evaluation metric?

3. **CKA as an online signal**: Has CKA been computed *during training* (not just at evaluation time) to drive adaptation decisions such as learning rate adjustment or gradient masking?

4. **Which XLM-R layers need the most work**: Is there existing empirical literature that characterizes which of XLM-R's 12 layers are most and least cross-lingually aligned (EN–ES or EN–ZH or other pairs) in the pretrained model, before any fine-tuning?

5. **Discriminative FT on multilingual models specifically**: Has the ULMFiT-style discriminative fine-tuning (depth-based LR scaling) been applied specifically to multilingual pretrained models (not just monolingual BERT/RoBERTa), and has its effect on cross-lingual transfer been studied?

6. **Alignment-weighted regularization**: Has anyone proposed a regularization scheme where the regularization strength at each layer is modulated by a data-driven measure of that layer's task relevance or alignment quality (rather than using uniform L2 weight decay across all layers)?

---

## Section 6: Exclusion Constraints

Please **exclude** the following to avoid noise:

- Papers focused solely on cross-lingual transfer for structured prediction (NER, dependency parsing, POS tagging) unless they explicitly study representation geometry or propose a fine-tuning method with geometric analysis
- Papers on language-specific fine-tuning (adapting a model to a single new language) unless the method generalizes to multilingual alignment improvement
- Neural machine translation papers that do not discuss alignment metrics on representation spaces (translation quality ≠ representation alignment)
- General survey / overview papers, unless the survey is specifically on cross-lingual alignment methods
- Papers from before 2018 (pre-BERT era), unless they introduce a metric or theoretical concept still widely used (e.g., Procrustes analysis, CKA)
- Papers on parameter-efficient fine-tuning (LoRA, adapters, prefix tuning) unless they specifically study cross-lingual alignment effects

---

## Section 7: Output Format for Each Paper Found

For each relevant paper, please use this exact format:

```
**Title:** [full paper title]
**Authors:** [First Author] et al.
**Venue/Year:** [e.g., ICLR 2024 / ArXiv 2023]
**ArXiv/DOI:** [link or ID if available]
**Relevance:** [1–2 sentences explaining why this is relevant to the adaptive cross-lingual fine-tuning project]
**Key contribution:** [1 sentence on the main technical novelty]
**Potential use in project:** [which specific module or design decision this paper informs — e.g., "alignment/scorer.py: suggests using RSA as an alternative to Procrustes" or "validates our hypothesis that middle layers of XLM-R are most aligned"]
```

At the end, please provide a brief summary of findings for each gap in Section 5 — specifically whether the gap remains open or has been addressed in the literature.
