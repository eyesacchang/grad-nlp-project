import torch
import torch.nn.functional as F


def infonce_loss(z_src: torch.Tensor, z_tgt: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss with in-batch negatives.

    Normalizes both embeddings, computes a B×B similarity matrix, and treats
    the diagonal as positive pairs. Loss is averaged over both directions.

    Args:
        z_src: (B, d) source language embeddings
        z_tgt: (B, d) target language embeddings
        temperature: tau scaling factor (lower = sharper distribution)

    Returns:
        scalar loss tensor
    """
    z_src = F.normalize(z_src, dim=-1)
    z_tgt = F.normalize(z_tgt, dim=-1)

    # Similarity matrix: (B, B), diagonal = positive pairs
    sim = z_src @ z_tgt.T / temperature

    labels = torch.arange(sim.size(0), device=sim.device)

    loss_src_to_tgt = F.cross_entropy(sim, labels)
    loss_tgt_to_src = F.cross_entropy(sim.T, labels)

    return (loss_src_to_tgt + loss_tgt_to_src) / 2.0
