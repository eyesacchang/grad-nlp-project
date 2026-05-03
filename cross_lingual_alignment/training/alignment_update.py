import torch
from typing import Dict, List, Tuple

from alignment.scorer import get_scorer
from alignment.scheduler import compute_lr_scales
from models.parameter_groups import update_param_group_lrs


def compute_alignment_scores(
    model,
    probe_pairs: List[Tuple[str, str]],
    collator,
    scorer_type: str,
    device,
) -> Dict[int, float]:
    """Compute per-layer alignment scores on a set of parallel sentence pairs.

    Args:
        model: XLMRWrapper (training model)
        probe_pairs: list of (src_text, tgt_text) pairs
        collator: PairCollator instance
        scorer_type: "procrustes" | "cosine" | "cka"
        device: torch device

    Returns:
        {layer_idx (1-indexed, 1-12): alignment_score}
    """
    scorer = get_scorer(scorer_type)
    model.eval()

    batch = collator(probe_pairs)
    input_ids_src = batch["input_ids_src"].to(device)
    attention_mask_src = batch["attention_mask_src"].to(device)
    input_ids_tgt = batch["input_ids_tgt"].to(device)
    attention_mask_tgt = batch["attention_mask_tgt"].to(device)

    with torch.no_grad():
        src_reps = model.get_all_layer_reps(input_ids_src, attention_mask_src)
        tgt_reps = model.get_all_layer_reps(input_ids_tgt, attention_mask_tgt)

    scores = {}
    for i, (src_rep, tgt_rep) in enumerate(zip(src_reps, tgt_reps)):
        layer_idx = i + 1  # 1-indexed to match parameter_groups convention
        scores[layer_idx] = scorer.score(src_rep.cpu(), tgt_rep.cpu())

    model.train()
    return scores


def maybe_update_lr(
    step: int,
    update_freq: int,
    model,
    optimizer,
    probe_pairs,
    collator,
    scorer_type: str,
    scheduler_type: str,
    scheduler_temperature: float,
    min_lr_scale: float,
    base_lr: float,
    device,
) -> Dict[int, float]:
    """Recompute alignment scores and update optimizer LRs every update_freq steps.

    Returns the current alignment scores (newly computed or empty dict if not updated).
    """
    if step % update_freq != 0:
        return {}

    scores = compute_alignment_scores(model, probe_pairs, collator, scorer_type, device)

    # Extend scores to include embeddings (0) and pooler (13) at neutral scale 0.5
    full_scores = {0: 0.5, **scores, 13: 0.5}

    lr_scales = compute_lr_scales(
        full_scores,
        scheduler_type=scheduler_type,
        temperature=scheduler_temperature,
        min_scale=min_lr_scale,
    )

    update_param_group_lrs(optimizer, lr_scales, base_lr)

    return scores
