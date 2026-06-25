import torch


@torch.no_grad()
def filtered_link_prediction(model, dataset, incidence, text_features, visual_features, triples, device):
    """Evaluate head and tail prediction with the filtered KGC protocol."""
    model.eval()
    # Encode once for evaluation because model parameters and input features are
    # fixed while ranking all candidate entities.
    ze, zr = model.encode(incidence, text_features, visual_features)
    ranks = []
    entities = torch.arange(dataset.num_entities, device=device)

    for h, r, t in triples:
        # Tail prediction: score (h, r, candidate) for every entity and mask
        # other known true tails before computing the target rank.
        h_batch = torch.full_like(entities, h)
        r_batch = torch.full_like(entities, r)
        scores, _, _ = model.reason_score(ze, zr, h_batch, r_batch, entities)
        for cand in range(dataset.num_entities):
            if cand != t and (h, r, cand) in dataset.all_true:
                scores[cand] = -1e9
        ranks.append(_rank(scores, t))

        # Head prediction mirrors the tail-ranking path.
        t_batch = torch.full_like(entities, t)
        scores, _, _ = model.reason_score(ze, zr, entities, r_batch, t_batch)
        for cand in range(dataset.num_entities):
            if cand != h and (cand, r, t) in dataset.all_true:
                scores[cand] = -1e9
        ranks.append(_rank(scores, h))

    ranks = torch.tensor(ranks, dtype=torch.float32)
    return {
        "MRR": float((1.0 / ranks).mean()),
        "Hits@1": float((ranks <= 1).float().mean()),
        "Hits@3": float((ranks <= 3).float().mean()),
        "Hits@10": float((ranks <= 10).float().mean()),
    }


def _rank(scores, target):
    # Optimistic tie handling matches common filtered KGC evaluation practice:
    # only strictly higher scores are counted ahead of the target.
    target_score = scores[target]
    return int((scores > target_score).sum().item()) + 1
