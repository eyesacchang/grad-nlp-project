# Adaptive Fine-Tuning for Cross-Lingual Alignment via Layer-Wise Geometry

Fine-tunes XLM-RoBERTa with per-layer adaptive learning rates driven by cross-lingual alignment quality. Poorly aligned layers receive larger updates; well-aligned layers receive smaller updates plus stronger regularization to prevent representational drift.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Smoke test (50 steps, no W&B)
python scripts/train.py --config configs/base_config.yaml --debug

# Full training run
python scripts/train.py --config configs/base_config.yaml

# Evaluate a checkpoint
python scripts/eval.py --checkpoint outputs/checkpoint_step5000.pt

# Run all baselines
python scripts/run_baselines.py --config configs/base_config.yaml

# Scrape related papers
python scripts/search_papers.py --output papers_found.csv
```

## Project Structure

```
configs/        hyperparameter configs and W&B sweep config
data/           OPUS-100 and FLORES-200 data loaders + collator
models/         XLM-R wrapper, parameter groups, pretrained snapshot
alignment/      alignment scorers (Procrustes, cosine, CKA), LR scheduler, hooks
losses/         InfoNCE contrastive loss, L2 regularization
training/       main trainer, adaptive LR update, logging callbacks
evaluation/     geometric eval (Procrustes, CKA, drift) and behavioral eval (P@k)
baselines/      pretrained eval, vanilla InfoNCE, static layer freeze
scripts/        train.py, eval.py, run_baselines.py, search_papers.py
notebooks/      exploratory analysis
```

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `alignment.scorer` | procrustes | Alignment metric driving LR scaling |
| `alignment.update_freq` | 500 | Steps between LR recomputation |
| `regularization.lambda_reg` | 0.1 | Set to 0 to ablate regularization |
| `loss.temperature` | 0.07 | InfoNCE temperature |
| `optimizer.base_lr` | 2e-5 | Base learning rate |

## Baselines

1. **Pretrained XLM-R** — no fine-tuning
2. **Vanilla InfoNCE** — uniform learning rate, no regularization
3. **Layer freeze** — bottom 6 layers frozen, top 6 trained at base_lr
