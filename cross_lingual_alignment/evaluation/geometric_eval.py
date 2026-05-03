import torch
from alignment.scorer import ProcrustesSimilarityScorer, CKAScorer, CosineSimilarityScorer


def run_geometric_eval(model, snapshot, eval_pairs, collator, device, max_pairs=1000):
    """Evaluate layer-wise geometric alignment and representational drift.

    Uses a fixed set of eval_pairs (up to max_pairs) for comparability across checkpoints.

    Returns dict of metrics prefixed with 'geo/'.
    """
    pairs = eval_pairs[:max_pairs]
    batch = collator(pairs)
    input_ids_src = batch["input_ids_src"].to(device)
    attention_mask_src = batch["attention_mask_src"].to(device)
    input_ids_tgt = batch["input_ids_tgt"].to(device)
    attention_mask_tgt = batch["attention_mask_tgt"].to(device)

    procrustes = ProcrustesSimilarityScorer()
    cka = CKAScorer()
    cosine = CosineSimilarityScorer()

    metrics = {}

    model.eval()
    with torch.no_grad():
        src_reps = model.get_all_layer_reps(input_ids_src, attention_mask_src)
        tgt_reps = model.get_all_layer_reps(input_ids_tgt, attention_mask_tgt)
        frz_src_reps = snapshot.get_all_layer_reps(input_ids_src, attention_mask_src)

    for i, (src_rep, tgt_rep, frz_rep) in enumerate(zip(src_reps, tgt_reps, frz_src_reps)):
        layer = i + 1
        X = src_rep.cpu()
        Y = tgt_rep.cpu()
        F_ = frz_rep.cpu()

        metrics[f"geo/procrustes_layer{layer}"] = procrustes.score(X, Y)
        metrics[f"geo/cka_layer{layer}"] = cka.score(X, Y)
        metrics[f"geo/cosine_layer{layer}"] = cosine.score(X, Y)
        # CKA between current and frozen (measures drift)
        metrics[f"geo/drift_cka_layer{layer}"] = cka.score(X, F_)

    model.train()
    return metrics
