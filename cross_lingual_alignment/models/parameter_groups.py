def build_parameter_groups(model, lr_scales, base_lr):
    """Build optimizer parameter groups with per-layer learning rates.

    Args:
        model: XLMRWrapper instance
        lr_scales: dict mapping layer_idx (0=embeddings, 1-12=encoder layers, 13=pooler)
                   to a float scale factor in [0, 1]
        base_lr: base learning rate (float)

    Returns:
        list of dicts suitable for passing to an optimizer
    """
    groups = []

    # Embedding layer -> scale index 0
    embedding_params = list(model.model.embeddings.parameters())
    if embedding_params:
        groups.append({
            "params": embedding_params,
            "lr": base_lr * lr_scales.get(0, 1.0),
            "name": "embeddings",
        })

    # Transformer layers 1-12 -> scale indices 1-12
    for i, layer in enumerate(model.model.encoder.layer):
        layer_params = list(layer.parameters())
        groups.append({
            "params": layer_params,
            "lr": base_lr * lr_scales.get(i + 1, 1.0),
            "name": f"encoder.layer.{i}",
        })

    # Pooler -> scale index 13
    if hasattr(model.model, "pooler") and model.model.pooler is not None:
        pooler_params = list(model.model.pooler.parameters())
        if pooler_params:
            groups.append({
                "params": pooler_params,
                "lr": base_lr * lr_scales.get(13, 1.0),
                "name": "pooler",
            })

    # Projection head — always at base_lr, never scaled by alignment scores.
    # It's randomly initialized (not pretrained), so alignment-based protection
    # doesn't apply; it needs a full LR to learn the contrastive projection.
    if hasattr(model, "projection"):
        groups.append({
            "params": list(model.projection.parameters()),
            "lr": base_lr,
            "name": "projection",
        })

    return groups


def update_param_group_lrs(optimizer, lr_scales, base_lr):
    """Update optimizer param group LRs in-place, preserving Adam momentum state.

    Expects optimizer.param_groups to have a 'name' field set by build_parameter_groups.
    """
    name_to_scale = {
        "embeddings": lr_scales.get(0, 1.0),
        "pooler": lr_scales.get(13, 1.0),
    }
    for i in range(12):
        name_to_scale[f"encoder.layer.{i}"] = lr_scales.get(i + 1, 1.0)

    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name in name_to_scale:
            group["lr"] = base_lr * name_to_scale[name]
