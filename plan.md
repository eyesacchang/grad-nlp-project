# Plan: Mechanistic ICL Ordering Sensitivity — Theory + Training Fix

## Context

The grad NLP project is pivoting from multilingual retrieval (Phase 2b language conditioning) to a fresh direction targeting a NeurIPS-workshop-quality paper. The new direction was selected after a three-wave literature sweep across NeurIPS 2024 / EMNLP 2024 / ACL 2025 / ICLR 2025 / arXiv 2024–2025 and a Wave 3 deep-dive, then validated by three independent agents (gap status, technical feasibility, methodological red-team).

**The contribution**: Use mechanistic interpretability (path patching) to identify the specific attention heads that cause few-shot in-context learning ordering sensitivity, derive a theory-grounded training objective from that diagnosis, and demonstrate the fix improves worst-case ordering accuracy on Llama-3.2-3B.

**Why this direction**: The phenomenon (ordering sensitivity, Zhao et al. 2021) is a well-documented decade-old empirical finding. Existing fixes are either (a) inference-time heuristics (Lu 2022, Batch-ICL, InvICL), (b) black-box training (PEARL adversarial DRO), or (c) architectural rewrites (Set-LLM permutation-invariant masking). **None provide a circuit-level causal account or a mechanistically-motivated training objective.** The team is making a one-shot bet, so confidence is grounded in agent-validated literature, toolchain, and methodology.

**The team has one attempt.** The plan therefore pre-registers hypotheses, requires a falsification criterion before training, and specifies a pivot if Phase 1 diagnosis fails to localize.

---

## Confidence Grounding

| Question | Status | Source |
|---|---|---|
| Is the gap still open as of 2026-05? | YES, MEDIUM confidence | Gap-validation agent: searched arXiv, ACL, OpenReview through April 2026; no paper combines mechanistic diagnosis + ordering-derived training objective |
| Biggest 2025 threat we missed? | **Set-LLM** (NeurIPS 2025, arXiv:2505.15433) — architectural permutation invariance via SetPE+SetMask | Gap-validation agent |
| Does TransformerLens support Llama-3.2? | YES, first-class for 1B and 3B | Verified in `loading_from_pretrained.py` and model_properties_table |
| Realistic compute on Modal? | $150–200 (not $20). A100 40GB is $2.10/hr | Modal pricing page + technical-feasibility agent |
| Critical methodology gap? | Random-head ablation must be in the experiment plan; without it, the "mechanistic targeting" claim is unfalsifiable | Red-team agent (Tier 1, #2) |

**Defensibility of contribution**: Three framings ranked by safety:
1. **(Aggressive)** "We find a head subset whose causal contribution *flips sign* under permutation, distinct from FV/induction heads" — strongest if Phase 1 reveals new heads.
2. **(Safe)** "First mechanistic-to-training pipeline targeting *variance under permutation* (vs. ABFT targeting correctness, PEARL targeting worst-case empirically, Set-LLM via architecture)."
3. **(Fallback)** "Diagnostic + lightweight LoRA regularizer competitive with heavier black-box DRO and architectural rewrites."

Plan to write the framing only after Phase 1 results come in.

---

## Pre-Registered Hypotheses

Lock these BEFORE running Phase 2 patching. Falsification criteria are mandatory.

**H1 (localization)**: Ordering sensitivity in Llama-3.2-1B/3B is causally driven by ≤10 attention heads concentrated in the upper half of the network (layers 8–16 for the 1B model; 14–28 for the 3B model). *Falsification*: if removing the top-10 heads by path-patching effect recovers <40% of ordering-induced accuracy variance, H1 is rejected.

**H2 (recency mechanism)**: The identified heads attend disproportionately to the last demonstration's label-token position relative to earlier label tokens, with attention weight on the last demo > 1.5× the uniform baseline (1/K). *Falsification*: if mean attention is within 1.2× of uniform across the identified heads, H2 is rejected and the "recency-bias" framing must be revised.

**H3 (training fix)**: A KL-divergence penalty applied to ONLY the identified heads (ASR-targeted) outperforms the same penalty applied to random heads of equal count (ASR-random) on min-accuracy across orderings, with effect size > 1pp at p<0.05 (paired bootstrap, 5 seeds). *Falsification*: if ASR-random matches ASR-targeted, the mechanistic-targeting contribution collapses; pivot to "PCT alone" framing.

**H4 (verification)**: After ASR-targeted training, the identified heads' attention weights toward the last demo's label position drop by ≥30% relative to baseline. *Falsification*: if attention patterns don't shift on the targeted heads, the training did not fix what we claim it fixed; report as null.

---

## Critical Methodology Decisions (locked)

These are the agent-validated answers to questions that will be asked at review.

| Decision | Choice | Reason |
|---|---|---|
| Patching technique | **Path patching** (Goldowsky-Dill 2023) + attribution patching as preliminary screen (Syed et al. NAACL 2024) | Activation patching across permuted inputs has upstream residual drift confounds. Attribution patching ranks heads in 3 passes, then path patching confirms top-K. |
| Primary metric | **Min-accuracy across all K! orderings** | Std can be gamed by uniformly bad models. Min-acc is what users actually care about and what reviewers will ask for. |
| Mean-accuracy guardrail | Pre-registered: mean accuracy may not drop more than 1pp vs. baseline | Otherwise the std/min-acc improvement could be from degraded representations. |
| Number of demonstrations K | **K=5** primary, K=3 secondary | K=3 is at the low end of what reviewers will accept; K=5 → 120 permutations gives statistical power. Sample 30 random permutations per instance, not all 120. |
| Random seeds | **5 seeds**: 42, 1337, 2024, 7, 31337 | Std is the metric; need paired bootstrap CIs. |
| Statistical test | Paired bootstrap over (instance, ordering) matrix; report 95% CIs | Within-instance correlated, must pair. |
| Models for diagnosis | **Pythia-2.8B + Llama-3.2-1B** (NOT GPT-2-XL) | Pythia has cleaner training dynamics for mech-interp; GPT-2-XL is noisier and obsolete. Both supported by TransformerLens. |
| Models for training | **Llama-3.2-3B** primary; **Qwen-2.5-1.5B** secondary | Single-model results = "doesn't replicate" objection. Two architectures shows generality. |
| Toolchain | **TransformerLens** for diagnosis; **HuggingFace + PEFT** for training with forward hooks (NOT `output_attentions=True`) | TL has off-the-shelf head patching utilities for Llama-3.2; eager attention from `output_attentions` kills SDPA throughput 2-3×. |
| LoRA targets | **q_proj, k_proj, v_proj, o_proj** (all four) | PEFT default is q_proj+v_proj only; insufficient for attention-pattern modification. |
| Variance loss | **Pairwise squared difference of output logits** across permutations, not variance over hidden states | Hidden-state variance can collapse trivially. |
| Tasks | **SST-2, AGNews, TREC, Subj** (classification) + **GSM8K** (generation, K=3) | Pure classification = "doesn't generalize" objection. |
| Mandatory baselines | Vanilla LoRA / GlobalE (Lu 2022) / Batch-ICL (Dong 2024) / **PEARL (reproduce on 1 task only)** / **Set-LLM (cite numbers)** / **ASR-random heads** / **best-ordering-only ceiling** | Skipping PEARL / Set-LLM / random-head is paper-killing. |
| Replication first | **First table in the paper**: ordering sensitivity replicated in Llama-3.2-1B/3B and Pythia-2.8B (range, std, min-acc on all 4 tasks) | Without this, the paper is built on a 2021 finding that may not hold on modern open LLMs. |

---

## Execution Plan (5 phases, ~6–8 weeks, $150–200)

### Phase 0 — Setup (3 days, $0)

1. Create fresh directory `icl_ordering/` outside the existing `cross_lingual_alignment/` package.
2. `pip install transformer_lens transformers peft datasets accelerate bitsandbytes` (lock versions in requirements.txt).
3. HuggingFace token, accept Llama-3.2 and Pythia licenses.
4. Smoke test: load Pythia-2.8B in TransformerLens on Colab T4, run a 3-shot prompt, verify logits match HF reference within 1e-3.
5. Set up Modal app with A100 40GB function for later phases.
6. **Verification**: `python -m icl_ordering.smoke_test` produces matching logits across TL and HF.

### Phase 1 — Replication of ordering sensitivity (1 week, ~$10)

1. Implement task loaders for SST-2, AGNews, TREC, Subj, GSM8K with class-balanced sampling.
2. Generate K=5 demonstration sets per test instance; sample 30 random permutations per instance.
3. Run vanilla ICL on Pythia-2.8B, Llama-3.2-1B, Llama-3.2-3B.
4. Produce **the first table of the paper**: per-model × per-task accuracy range, std, min-acc, max-acc across permutations.
5. **Gate**: at least 3 of (model, task) cells must show >5pp range. If not, ordering sensitivity has weakened on modern LLMs and the paper premise needs revision.

### Phase 2 — Mechanistic diagnosis (1.5 weeks, ~$20)

1. **Step 2a (Attribution patching screen)**: For 200 instances per task, run attribution patching across all heads on Llama-3.2-1B (24 layers × 32 Q heads = 768 heads). Three forward+backward passes per instance. Outputs an effect score per head.
2. **Step 2b (Path patching confirmation)**: Take top-15 heads by attribution score. Run path patching for each (target: final logit; source: best-ordering vs worst-ordering activation at each head's output). Specify token position (last demo label position), component (attention pattern vs OV output), and receiver (final layernorm input).
3. **Step 2c (H1 test)**: Confirm that ablating top-10 heads recovers ≥40% of ordering-induced accuracy variance. If not, H1 rejected — pivot to layer-block-level theory (still publishable).
4. **Step 2d (H2 test)**: For each of the top-10 heads, measure mean attention weight on last demo's label position vs uniform (1/K). Need >1.5× ratio for H2. If not, revise framing from "recency bias" to whatever the patching reveals.
5. **Step 2e (FV head differentiation)**: Compare identified heads to Function Vector heads (Kim et al. 2502.14010). If they overlap >70%, the contribution is confirmation not discovery — switch to framing (B) "first training-fix targeting these heads" rather than "we found new heads."
6. **Lock the head set**. Save `identified_heads.json` with layer/head indices. No revisions after this point.

### Phase 3 — Training (2 weeks, ~$80)

Two objectives, each trained with LoRA r=8 on q/k/v/o projections, 2K examples × 4 classification tasks × 5 seeds.

**ASR-targeted**: Forward hook on `self_attn` of identified heads only. Compute KL(softmax(attn_pattern[query_to_demo_label_positions]) || Uniform(K)). Add λ·KL to loss. λ ∈ {0.1, 0.5, 1.0}; tune on Subj dev.

**ASR-random** (control): Same as ASR-targeted but applied to a random set of heads of equal count. Same λ. **This is the falsification control for H3.**

**PCT**: For each batch, sample 3 permutations of the K demonstrations. Forward each. Compute pairwise squared difference of output logits, sum, add λ_pct·(sum/3) to loss. λ_pct ∈ {0.1, 0.5, 1.0}.

**ASR-targeted + PCT**: Combined.

Total runs: 4 objectives × 4 tasks × 5 seeds = 80 runs at ~30min each on A100. Plus Llama-3.2-3B retrained on Qwen-2.5-1.5B for the secondary model: another 80 runs. **Use only 2 hyperparams** (low/high λ) for the secondary model to save cost.

### Phase 4 — Evaluation (1 week, ~$30)

For each trained model:
1. Run all K!=120 orderings on 500 held-out test instances per task. Compute mean, min, std accuracy.
2. Paired bootstrap (n=10000) over (instance, ordering) matrix; 95% CIs.
3. Compare to baselines: vanilla LoRA on each task, GlobalE, Batch-ICL, PEARL on SST-2 only (single reproduction), Set-LLM cited numbers, best-ordering-only ceiling.
4. **GSM8K transfer evaluation**: K=3 demos, 50 random permutations on 200 problems. Report exact-match min-acc.
5. **Leave-one-task-out**: train on 3 classification tasks, test on the 4th. Verifies the fix isn't task-specific overfitting.

### Phase 5 — Mechanistic verification (3 days, ~$10)

1. Re-run path patching on the ASR-targeted model on 200 instances per task.
2. Test H4: identified heads' attention to last-demo-label-position dropped ≥30%? Report effect size with bootstrap CI.
3. **Discriminating control**: re-run path patching on ASR-random model. The identified heads should NOT have changed in ASR-random training; if they did, the targeting was incidental.
4. Produce the verification figure: attention pattern heatmap before/after for the top-5 identified heads, side-by-side with random-head control.

---

## Pivot Plans (if hypotheses fail)

| Failure | Pivot |
|---|---|
| Phase 1 gate fails (no ordering sensitivity in modern open LLMs) | Reframe paper as "ordering sensitivity in modern LLMs has been mitigated by RLHF/instruction-tuning — diagnostic study + characterization." Still publishable at a workshop. |
| H1 fails (no clean head-level localization) | Pivot to layer-block-level theory. Use ASR at the layer level. Frame as "ordering sensitivity is distributed; targeted layer-block regularization beats global." |
| H2 fails (mechanism is not recency) | Adjust framing to whatever the patching reveals. Still novel. |
| H3 fails (random heads match targeted) | Pivot to "PCT-only" framing: theory-derived objective is no better than consistency training, but consistency training is itself a strong, simple baseline. Mech-interp section becomes "diagnostic" rather than "load-bearing." |
| H4 fails (training didn't change targeted heads) | Report null. Frame paper as "ASR improves accuracy via mechanism we cannot localize," demote to short paper. |
| PCT alone matches PCT+ASR | Strong negative result — train only on PCT. Cleaner paper, stronger ablation story. |

**Each pivot has been pre-considered and is publishable.** This is the insurance against the one-shot constraint.

---

## Compute Budget (validated)

| Phase | A100 hours | Cost |
|---|---|---|
| Phase 0 setup + smoke tests | 1 hr | $2 |
| Phase 1 replication | 5 hr | $10 |
| Phase 2 diagnosis | 10 hr | $21 |
| Phase 3 training (Llama 3B + Qwen 1.5B, 160 runs) | 50 hr | $105 |
| Phase 4 evaluation (120 perms × 5 models × 4 tasks) | 12 hr | $25 |
| Phase 5 verification | 4 hr | $9 |
| Buffer / debugging | 15 hr | $32 |
| **Total** | **97 hr** | **~$200** |

A100 40GB Modal pricing: $2.10/hr (verified 2026-05). Use Colab T4 for diagnosis preliminaries to save Modal hours.

---

## Critical Files to Create

```
icl_ordering/
├── data/
│   ├── loaders.py            # SST-2, AGNews, TREC, Subj, GSM8K with class-balanced sampling
│   └── permutations.py       # K-permutation generator with deterministic seeding
├── diagnosis/
│   ├── attribution_patch.py  # screen all heads via Syed et al. attribution patching
│   ├── path_patch.py         # confirm top-15 via path patching
│   └── identified_heads.json # locked output of Phase 2
├── training/
│   ├── asr.py                # Attention Symmetry Regularization (forward hook implementation)
│   ├── pct.py                # Permutation Consistency Training
│   ├── lora_train.py         # main training loop
│   └── configs/              # YAML configs per objective × model
├── eval/
│   ├── full_permutation_eval.py  # all K!=120 orderings
│   ├── bootstrap.py              # paired bootstrap CIs
│   └── leave_one_task_out.py
├── verification/
│   └── post_train_patching.py  # H4 test
├── baselines/
│   ├── globale.py
│   ├── batch_icl.py
│   └── pearl_repro.py        # reproduce on SST-2 only
├── analysis/
│   ├── attention_heatmaps.py
│   └── results_aggregation.py
└── README.md
```

**Files to not touch**: `cross_lingual_alignment/`, `phase 2/`, `phase 3/`, `iso_colbert.ipynb`. The new direction is independent of the prior multilingual work.

---

## Verification Checklist (before claiming done)

- [ ] Phase 1 table replicates Zhao 2021 in modern open LLMs
- [ ] H1 confirmed (or H1 rejected → layer-block pivot)
- [ ] H2 confirmed (or framing revised)
- [ ] FV-head differentiation analysis included
- [ ] H3: ASR-targeted vs ASR-random comparison with paired bootstrap CI
- [ ] H4: post-training attention patterns measured on identified + random heads
- [ ] All 5 baselines run, including PEARL on SST-2
- [ ] GSM8K transfer numbers reported
- [ ] Leave-one-task-out numbers reported
- [ ] 5 seeds × 4 tasks × 4 objectives × 2 models complete
- [ ] Mean-accuracy guardrail not violated (≤1pp drop)
- [ ] Pivot section written in paper for any failed hypothesis

---

## Risks Acknowledged

1. **Set-LLM threat** — reviewers will ask "why a soft regularizer beats architectural invariance?" Answer: ASR is parameter-efficient (LoRA), retains general ICL ability, doesn't require architectural changes that break HF compatibility. Mention explicitly in related work.
2. **FV heads overlap** — if our identified heads overlap heavily with Kim et al.'s FV heads, contribution is confirmation. Pivot to framing (B).
3. **3B model "too small"** — having Qwen-2.5-1.5B as second model, plus K=5 evaluation, plus GSM8K, blunts this objection. If reviewers still push, validate on Llama-3.2-3B-Instruct as a robustness check (~$10 extra).
4. **Compute overrun** — buffer is 15hr ($32). If Phase 3 blows budget, drop the secondary model (Qwen) hyperparameter sweep first; keep one λ value.
