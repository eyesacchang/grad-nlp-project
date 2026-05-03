import torch
import torch.nn.functional as F


def run_behavioral_eval(model, eval_pairs, collator, device, batch_size=256):
    """Cross-lingual sentence retrieval on parallel pairs.

    For each source sentence, ranks all target sentences by cosine similarity
    and reports P@1, P@5, P@10 (precision at k).

    Returns dict of metrics prefixed with 'eval/'.
    """
    model.eval()
    all_src, all_tgt = [], []

    for start in range(0, len(eval_pairs), batch_size):
        batch_pairs = eval_pairs[start:start + batch_size]
        batch = collator(batch_pairs)
        input_ids_src = batch["input_ids_src"].to(device)
        attention_mask_src = batch["attention_mask_src"].to(device)
        input_ids_tgt = batch["input_ids_tgt"].to(device)
        attention_mask_tgt = batch["attention_mask_tgt"].to(device)

        with torch.no_grad():
            z_src = model(input_ids_src, attention_mask_src)
            z_tgt = model(input_ids_tgt, attention_mask_tgt)

        all_src.append(z_src.cpu())
        all_tgt.append(z_tgt.cpu())

    all_src = torch.cat(all_src, dim=0)  # (N, d)
    all_tgt = torch.cat(all_tgt, dim=0)  # (N, d)

    # Cosine similarity matrix: (N, N)
    sim_matrix = all_src @ all_tgt.T  # already L2-normalized from wrapper

    N = sim_matrix.size(0)
    labels = torch.arange(N)

    def precision_at_k(k):
        topk_indices = sim_matrix.topk(k, dim=1).indices  # (N, k)
        correct = (topk_indices == labels.unsqueeze(1)).any(dim=1)
        return correct.float().mean().item()

    metrics = {
        "eval/p_at_1": precision_at_k(1),
        "eval/p_at_5": precision_at_k(5),
        "eval/p_at_10": precision_at_k(10),
        "eval/mean_pos_cosine": sim_matrix.diagonal().mean().item(),
    }

    model.train()
    return metrics
