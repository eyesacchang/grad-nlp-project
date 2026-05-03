import torch
import math


def compute_lr_scales(alignment_scores: dict, scheduler_type="softmax_inverse",
                      temperature=1.0, min_scale=0.01) -> dict:
    """Convert per-layer alignment scores to per-layer LR scale factors.

    Args:
        alignment_scores: {layer_idx: score} where score ∈ [0, 1], higher = better aligned
        scheduler_type: "linear" | "softmax_inverse" | "exponential"
        temperature: controls sharpness of differentiation across layers
        min_scale: floor for all scale values (prevents complete gradient blocking)

    Returns:
        {layer_idx: lr_scale} where lr_scale ∈ [min_scale, 1.0]
    """
    layer_ids = sorted(alignment_scores.keys())
    scores = torch.tensor([alignment_scores[i] for i in layer_ids], dtype=torch.float32)

    if scheduler_type == "linear":
        # Directly invert: poorly aligned → high LR
        raw_scales = 1.0 - scores

    elif scheduler_type == "softmax_inverse":
        # Invert scores, apply softmax with temperature, then rescale so mean = 1
        inv_scores = 1.0 - scores
        softmax_scales = torch.softmax(inv_scores / temperature, dim=0)
        raw_scales = softmax_scales * len(layer_ids)  # mean ≈ 1

    elif scheduler_type == "exponential":
        # exp(-k * score), sharpness controlled by 1/temperature
        k = 1.0 / max(temperature, 1e-9)
        raw_scales = torch.exp(-k * scores)

    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # Normalize to [0, 1] then apply floor
    max_val = raw_scales.max().clamp(min=1e-9)
    normalized = raw_scales / max_val
    clamped = normalized.clamp(min=min_scale, max=1.0)

    return {layer_id: float(clamped[i].item()) for i, layer_id in enumerate(layer_ids)}
