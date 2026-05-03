import torch
from typing import List, Optional, Dict


def representation_drift_loss(
    current_reps: List[torch.Tensor],
    frozen_reps: List[torch.Tensor],
    alignment_scores: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    """L2 regularization loss penalizing drift from pretrained representations.

    Per-layer loss = ||current_rep - frozen_rep||_F^2 / (B * d), normalized by
    batch size and hidden dim to be scale-invariant.

    Default is uniform weight across all layers. Passing alignment_scores enables
    alignment-weighted regularization (ablation only) — but this is redundant with
    the adaptive LR mechanism and should not be the default: both signals would
    protect well-aligned layers, double-counting the alignment information.

    Args:
        current_reps: list of (B, d) tensors from the current (training) model
        frozen_reps: list of (B, d) tensors from the frozen pretrained snapshot
        alignment_scores: optional {layer_idx (0-indexed): score ∈ [0,1]}.
                          Only pass this for ablation experiments.

    Returns:
        scalar loss tensor (mean of per-layer losses, uniform by default)
    """
    assert len(current_reps) == len(frozen_reps), "Layer count mismatch"
    B, d = current_reps[0].shape

    total_loss = torch.tensor(0.0, device=current_reps[0].device)

    for i, (cur, frz) in enumerate(zip(current_reps, frozen_reps)):
        per_layer = (cur - frz).pow(2).sum() / (B * d)
        weight = alignment_scores.get(i, 1.0) if alignment_scores is not None else 1.0
        total_loss = total_loss + weight * per_layer

    # Mean across layers so lambda_reg is invariant to model depth
    return total_loss / len(current_reps)
