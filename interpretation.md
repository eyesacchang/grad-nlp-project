# Experiment Notebooks: Interpretations

---

### Phase 1: Post-Alignment Geometry
`phase 1/post_alignment_geometry.ipynb`

After fine-tuning XLM-R on English-Spanish sentence pairs with InfoNCE, the model gets very good at retrieving Spanish translations of English queries, but only when the candidate pool contains Spanish sentences exclusively. This notebook asks: what happens when you expand the pool to include French, German, Swahili, and Arabic translations of the same sentences? And can cheap post-hoc fixes (whitening, margin scoring) close the gap?

**What it does:**
- Compares easy-pool P@1 (1012 Spanish candidates) vs hard-pool P@1 (5060 candidates across 5 languages)
- Applies whitening and margin scoring to the embeddings, measures recovery
- Checks for hub structure (the geometry pathology these fixes are designed to treat)
- Decomposes every hard-pool failure into: retrieved the right sentence in the wrong language vs retrieved an unrelated sentence
- Compares the raw 768-dim encoder space vs the 256-dim projected space InfoNCE trained directly
- Tests the same model on OPUS-100 (non-parallel benchmark) to check if the failure is benchmark-specific
- Runs pairwise two-language pools (ES+PT, ES+FR, ES+DE, ES+SW, ES+AR) to quantify the proximity gradient

**Results:**
- Easy pool P@1 = 0.994. Hard pool P@1 = 0.382. The gap is 0.612.
- Post-hoc fixes make things worse, not better: whitening costs 0.039 P@1 points, margin scoring costs 0.020, and both together cost 0.060.
- Hub fraction = 0.000. There is no geometry pathology. These fixes are solving a problem that does not exist here.
- P@1-any = 1.000. Every single hard-pool failure is a wrong-language true positive. The model always retrieves the right sentence, just often in French or German instead of Spanish.
- FR accounts for 53% of errors, DE for 45%, SW and AR together under 2%.
- The projected space (gap = 0.667) is worse than the encoder space (gap = 0.612). InfoNCE actively degrades language discrimination rather than just ignoring it.
- OPUS-100 gap = 0.013 vs FLORES gap = 0.612. The failure is invisible on standard benchmarks because competing-language translations are never in the candidate pool.
- Proximity gradient: ES+PT = 0.364 (worst), ES+FR = 0.567, ES+DE = 0.542, ES+SW = 0.936, ES+AR = 0.902. Surface similarity matters more than language family membership. German confuses nearly as badly as French despite being Germanic.

---

### Phase 1: Training Curve
`phase 1/training_curve.ipynb`

One natural explanation for the hard-pool gap is undertraining: maybe the model just needs more steps to learn that Spanish and French are different. This notebook tests that by training to 20K steps and measuring P@1-target and P@1-any at 2K, 5K, 10K, and 20K.

**What it does:**
- Trains vanilla InfoNCE on English-Spanish OPUS-100 bitext to 20K steps
- Evaluates on the FLORES hard pool at four checkpoints
- Checks whether P@1-target closes toward P@1-easy over time, or whether the gap is structural

**Results:**
- P@1-any = 1.000 at every checkpoint. The model always finds the right sentence.
- P@1-target starts at 0.382 at 2K steps, drops to 0.259 at 10K, and partially recovers to 0.297 at 20K.
- The gap grows from 0.612 to 0.738 before settling at 0.701. More training makes things worse.
- Undertraining is ruled out. The problem is structural: InfoNCE has no loss term for language identity, so additional training just collapses cross-lingual representations further, eroding whatever language specificity the pretrained model had.

---

### Phase 2a: Hard Negative Mining
`phase 2/hard_neg_experiment.ipynb`

If the problem is that InfoNCE never sees wrong-language translations as negatives, the minimal fix is to inject them. During training, for each English query in a FLORES batch, the French, German, Swahili, and Arabic translations of the same sentence are added as hard negatives in the InfoNCE denominator. No architecture change required.

**What it does:**
- Trains XLM-R with standard InfoNCE on OPUS-100 plus an auxiliary hard negative term on FLORES dev 6-tuples
- FR and DE negatives are weighted 2x (matching the observed error distribution)
- Evaluates hard-pool P@1 at checkpoints and runs a bystander eval: for each language, how well can the model still retrieve in that language when it is the desired target?
- Tests OPUS-100 EN-ES retrieval as a regression check

**Results:**
- Hard negative mining closes 99.7% of the FLORES gap in 2000 steps. P@1-target goes from 0.382 to 0.991.
- The fix is very fast: the hard neg loss drops from 1.80 to 0.02 between steps 200 and 400. The model had the representational capacity all along and just needed the signal.
- Bystander collapse: EN-to-French P@1 drops to 0.020, EN-to-German to 0.013, EN-to-Swahili to 0.016, EN-to-Arabic to 0.155. On a 1012-candidate pool where chance is 0.1%, FR/DE/SW are essentially at chance.
- OPUS-100 EN-ES regression: 0.890 vs 0.987 for vanilla, a 10% drop on standard bilingual retrieval.
- The problem is the global push: the loss tells the model that EN queries should never retrieve French or German, and the model generalizes this unconditionally to all EN-to-FR/DE retrieval. It is a specialized fix for fixed-target-language systems, not a general multilingual retrieval improvement.

---

### Phase 2b: Language-Conditioned Query Encoder
`phase 2/lang_conditioned_encoder_experiment.ipynb`

Hard negative mining works by globally repelling bystander languages during training. Phase 2b replaces that with a targeted pull at inference time. A learned per-language embedding is summed into the unit-normalized mean-pooled representation before the projection head. At inference, the query is conditioned on the desired target language, shifting it toward that language's subspace without affecting how bystander languages are treated during training.

**What it does:**
- Adds a learned language embedding table (5 languages, 768-dim each) to the encoder
- The mean pool is normalized to unit norm before the language embedding is added, so the embedding contributes roughly 27% of the combined vector rather than being swamped by the raw pool magnitude
- Trains with standard InfoNCE on OPUS-100 plus an auxiliary 6-tuple contrastive loss on FLORES dev that pulls conditioned queries toward their target-language translations and pushes away other-language translations of the same sentence
- Sweeps the auxiliary loss weight (lambda) over 0.1, 0.5, 1.0, and 2.0
- Evaluates hard-pool P@1 with ES-conditioned queries, and bystander easy-pool P@1 with each language conditioned on its own embedding

**Results:**
- Lambda = 0.5 is best: P@1-target = 0.9802, closing 97.6% of the FLORES gap.
- The lambda-performance relationship is non-monotonic: too small (0.1) and the language signal is too weak; too large (2.0) and the auxiliary loss interferes with the main InfoNCE objective.
- No bystander collapse: when EN queries are conditioned on French, French retrieval P@1 = 0.996. German = 0.999, Swahili = 0.948, Arabic = 0.973. All near ceiling.
- The model never learns a global rule against retrieving bystander languages. It learns that a query conditioned on "French" should retrieve French, and a query conditioned on "Spanish" should retrieve Spanish. These are different conditioning directions.
- Trade-off: the query encoder needs to know the target language at inference time. For multilingual RAG where the corpus language is known in advance, this is not a meaningful constraint.

---

### Phase 2: RAG Validation Experiment
`phase 2/rag_validation_experiment.ipynb`

The retrieval experiments establish that vanilla InfoNCE collapses language identity in the encoder, and that Phase 2b fixes this. This notebook connects the retrieval failure to actual downstream LLM behavior: does wrong-language retrieval cause a language model to generate in the wrong language, even when instructed to answer in Spanish?

**What it does:**
- Uses XQuAD, a reading comprehension dataset where English questions, Spanish passages, German passages, and Arabic passages are parallel translations of the same 1190 examples
- Builds a 750-passage corpus (250 Spanish, 250 German, 250 Arabic) and queries it with English questions targeting Spanish answers
- Runs retrieval with five encoders: pretrained XLM-R, LaBSE, vanilla InfoNCE 2K, hard neg 2K, and Phase 2b (lambda=0.5)
- Passes the top-1 retrieved passage to GPT-4o mini with the instruction to answer in Spanish, then measures what language the output is actually in
- Reports retrieval language accuracy (fraction of top-1 passages in Spanish) and answer language drift (fraction of GPT outputs not in Spanish)

**Results:**
- Not yet run. Expected: retrieval language accuracy tracks FLORES P@1-target across models, and answer language drift is inversely correlated with retrieval language accuracy, confirming that fixing the encoder eliminates the need for decoder-side workarounds.
