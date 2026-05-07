# Project Plan: Language Identity Collapse in Multilingual Retrieval

**Venue:** MRL Workshop @ EMNLP 2025  
**Deadline:** August 23, 2025  
**Team:** isaacchang@g.harvard.edu, ingridchien@g.harvard.edu, mtran@mit.edu

---

## The Problem

Multilingual RAG systems suffer from **language drift**: a user queries in English targeting a Spanish knowledge base, but the retriever returns French or German documents instead — because semantically equivalent translations are indistinguishable in embedding space. The LLM then generates in the wrong language. This failure is documented in production (arXiv:2511.09984).

Prior work patches it at the decoder side (Soft Constrained Decoding, DKM-RAG). We fix the root cause: **bilingual InfoNCE training systematically destroys language identity in sentence encoders**, and standard benchmarks are structurally blind to it because they never put competing-language candidates in the same retrieval pool.

---

## What We Know (Phase 1 + Phase 2a Complete)

### Core numbers

| Model | P@1-easy | P@1-target | P@1-any | Gap |
|---|---|---|---|---|
| Pretrained XLM-R (no FT) | 0.721 | 0.510 | 0.807 | 0.211 |
| LaBSE (zero-shot) | 1.000 | 0.175 | 1.000 | 0.825 |
| Vanilla InfoNCE 2K steps | 0.994 | 0.382 | 1.000 | 0.612 |
| Vanilla InfoNCE 20K steps | 0.998 | 0.297 | 1.000 | 0.701 |
| Vanilla 2K + langdetect filter | 0.994 | 0.991 | 0.991 | 0.003 |
| Oracle language filter | 0.994 | 0.994 | 0.994 | 0.000 |
| Hard neg mining 2K steps | 0.993 | 0.991 | 0.993 | 0.002 |

**P@1-target** = right sentence, right language. **P@1-any** = right sentence regardless of language. The gap is the failure.

### Phase 1 findings

- **Every failure is a wrong-language true positive.** P@1-any = 1.000 throughout all training.
- **More InfoNCE training makes it worse.** Gap grows monotonically from 2K to 10K steps.
- **Severity follows typological proximity.** PT = 0.364 < FR = 0.567 ≈ DE = 0.542 << SW = 0.936 ≈ AR = 0.902.
- **Standard benchmarks are blind to it.** OPUS-100 gap = 0.013, FLORES gap = 0.612, same model. OPUS-100 is non-multiway — wrong-language candidates never enter the pool.
- **The failure is encoder-deep.** 768-dim encoder gap ≈ 256-dim projected gap — the projection head compounds it slightly rather than correcting it.

### Phase 2a: hard negative mining works — but only as a specialized fix

Hard neg mining closes **99.7% of the FLORES gap** in 2000 steps. But the bystander eval reveals it is not a general fix:

**Bystander easy-pool eval (EN→{lang} P@1, hard neg 2K vs vanilla 2K):**

| Lang | Hard Neg 2K | Vanilla 2K | Δ |
|---|---|---|---|
| ES (target, control) | 0.995 | 0.994 | +0.001 |
| FR | 0.020 | ~0.97 | ~−0.95 |
| DE | 0.013 | ~0.97 | ~−0.96 |
| SW | 0.016 | ~0.94 | ~−0.92 |
| AR | 0.155 | ~0.88 | ~−0.73 |

**OPUS-100 EN-ES test P@1:** hard neg 2K = 0.890 vs vanilla 2K = 0.987 — a ~10% regression on standard bilingual retrieval.

FR, DE, and SW collapse to ~1-2% P@1 on a 1012-candidate pool where chance is 0.1%. The model has learned "EN queries must never retrieve French/German/Swahili" with no nuance — a global push that trades one failure mode for a worse one for any use case beyond fixed-target-language retrieval. The 10% OPUS-100 regression shows the hard neg loss also distorts the EN-ES embedding relationship itself.

**Implication for the paper:** Hard neg mining is viable only for single-target-language systems. This makes Phase 2b — a language-conditioned encoder that shifts the query toward the target at inference time rather than globally repelling bystanders during training — not merely incremental, but the **central contribution**.

---

## Remaining Experiments

### 1. Phase 2b: language-conditioned query encoder *(central contribution)*
**Architecture:** learned language embedding (dim 768) summed into mean-pooled CLS before the projection head. Five embeddings: ES, FR, DE, SW, AR. At inference, the query is conditioned on the target language — a targeted pull rather than a global push.  
**Loss:** standard InfoNCE on OPUS-100 + auxiliary contrastive loss on FLORES dev 6-tuples, with FR/DE negatives weighted 2×.  
**Sweep:** lambda_aux ∈ {0.1, 0.5, 1.0, 2.0}.  
**Eval:** same hard pool P@1-target/any as Phase 1 + **bystander easy-pool eval** (must show FR/DE/SW/AR do not collapse) + OPUS-100 regression check.

New files:
- `phase 2/models/language_conditioned_wrapper.py`
- `phase 2/losses/language_contrastive.py`
- `phase 2/phase2b_experiment.ipynb`

### 2. RAG validation experiment *(connects retrieval finding to production failure)*
End-to-end demonstration showing that fixing the encoder directly reduces language drift in a full RAG pipeline.

**Setup:**
- Dataset: MKQA or XQuAD (English questions, answers verifiable multilingually)
- Corpus: Wikipedia passages in ES + FR + DE + SW + AR on overlapping topics
- Query: English questions targeting Spanish answers
- Encoders tested: pretrained XLM-R, LaBSE, vanilla InfoNCE 2K, hard neg 2K, Phase 2b

**Metrics:**
- *Retrieval language accuracy* — fraction of top-1 retrieved passages in Spanish
- *Answer language drift* — pass retrieved passage to GPT-4o mini, measure output language
- *Answer correctness* — does fixing language drift maintain QA accuracy?

New file: `phase 2/rag_validation_experiment.ipynb`

---

## How the Two Fixes Compare

| | Hard Neg Mining | Language-Conditioned Encoder |
|---|---|---|
| Mechanism | Global push: bystander languages away from EN | Targeted pull: query toward target-language space |
| FLORES gap closed | 99.7% | TBD |
| Bystander cost | Catastrophic (FR/DE/SW → ~1-2%) | Expected none |
| OPUS-100 regression | ~10% | Expected minimal |
| Inference overhead | None | Requires knowing target language at query time |
| Architecture change | None | One language embedding layer |
| Use case | Fixed-target-language retrieval only | General multilingual RAG |
| Prior art | Known technique; novel application | Novel for cross-lingual retrieval |

The paper reports all four conditions (vanilla, hard neg, Phase 2b, combined) with the bystander eval as the key differentiator.

---

## Novelty

1. **New failure mode.** First paper to identify the P@1-target / P@1-any gap as a distinct failure mode of bilingual contrastive training and characterize its typological proximity gradient.
2. **Benchmark critique.** Standard monolingual-pool benchmarks (OPUS-100, Tatoeba) structurally conceal this failure — wrong-language candidates are simply never in the pool.
3. **Production link.** Directly connects the retrieval failure to documented language drift in multilingual RAG and fixes it at the root cause (encoder training) rather than with decoder-side patches.
4. **Architecture.** Language-conditioned query encoder is novel for cross-lingual retrieval. Unlike hard neg mining (which is a known technique and provably not general), it avoids bystander degradation via targeted inference-time conditioning. No prior work applies this to language in retrieval (confirmed; closest: CASE, Hyper-CL — neither applied to language in retrieval).

---

## Related Work

- **Language Drift in Multilingual RAG** (arXiv:2511.09984) — documents the production failure we fix at the root cause
- **Modular Sentence Encoders**, Huang et al. ACL 2025 (arXiv:2407.14878) — closest prior work on language-specific modules
- **LAReQA**, Roy et al. EMNLP 2020 — closest benchmark design; rewards language-agnostic retrieval (opposite goal to ours)
- **CASE**, Condition-Aware Sentence Embeddings (EACL 2026) — prior art for conditioned embeddings
- **LANGSAE EDITING** (arXiv:2601.04768) — inverse approach: removes language signal post-hoc
- **LaBSE**, Feng et al. ACL 2022 — language-agnostic baseline (collapses harder than vanilla InfoNCE)
- **ColBERT-XM** (arXiv:2402.15059) — modular multilingual retrieval, different architecture
- **Artetxe & Schwenk LASER 2019** — margin scoring baseline we test as post-hoc fix

---

## Next Steps

1. Implement and train Phase 2b language-conditioned encoder — run hard pool + bystander + OPUS-100 eval
2. Design and run RAG validation experiment
3. Write paper
