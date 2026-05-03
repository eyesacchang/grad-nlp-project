from contextlib import contextmanager
import torch


@contextmanager
def layer_rep_hooks(model, attention_mask):
    """Context manager that registers forward hooks to capture all 12 layer outputs.

    Usage:
        with layer_rep_hooks(xlmr_wrapper.model, attention_mask) as layer_reps:
            _ = xlmr_wrapper.model(input_ids, attention_mask)
        # layer_reps is now a dict {layer_idx (0-11): (B, d) mean-pooled tensor}

    Note: prefer xlmr_wrapper.get_all_layer_reps() for most use cases.
    This context manager is useful when you need reps mid-forward-pass.
    """
    captured = {}
    handles = []

    def make_hook(layer_idx, mask):
        def hook(module, input, output):
            hidden = output[0]  # transformer layer output is a tuple; first elem is hidden states
            m = mask.unsqueeze(-1).float()
            summed = (hidden * m).sum(dim=1)
            counts = m.sum(dim=1).clamp(min=1e-9)
            rep = torch.nn.functional.normalize(summed / counts, dim=-1)
            captured[layer_idx] = rep
        return hook

    for i, layer in enumerate(model.encoder.layer):
        handle = layer.register_forward_hook(make_hook(i, attention_mask))
        handles.append(handle)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
