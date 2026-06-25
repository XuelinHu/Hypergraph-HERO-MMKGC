import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hero import HERO, KGDataset
from hero.evaluate import filtered_link_prediction
from hero.utils import batch_iter, load_json, resolve_device, set_seed


def main():
    """Run the two-stage HERO training pipeline from a JSON configuration file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    set_seed(cfg["train"].get("seed", 42))
    device = resolve_device(cfg["train"].get("device", "cuda"))

    dataset = KGDataset(cfg["dataset"]["root"])
    incidence = dataset.build_incidence().to(device)
    text = dataset.text_features.to(device) if dataset.text_features is not None else None
    visual = dataset.visual_features.to(device) if dataset.visual_features is not None else None
    text_dim = 0 if text is None else text.shape[1]
    visual_dim = 0 if visual is None else visual.shape[1]

    model = HERO(
        dataset.num_entities,
        dataset.num_relations,
        text_dim,
        visual_dim,
        **cfg["model"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Entities={dataset.num_entities}, Relations={dataset.num_relations}, Train={len(dataset.train)}")

    # Stage 1 follows the manuscript's MKGPH pre-training objective:
    # triple discrimination plus hyperedge-level contrastive learning.
    print("Stage 1: MKGPH pre-training")
    for epoch in range(1, cfg["train"]["pretrain_epochs"] + 1):
        model.train()
        total = 0.0
        samples = dataset.negative_sample(dataset.train, cfg["train"]["negative_ratio"])
        for batch in tqdm(batch_iter(samples, cfg["train"]["batch_size"]), desc=f"pretrain {epoch}", leave=False):
            h, r, t, y = dataset.as_tensors(batch, device)
            ze, zr = model.encode(incidence, text, visual)
            score = model.pretrain_score(ze, zr, h, r, t)
            ht_loss = F.binary_cross_entropy_with_logits(score, y)
            cl_loss = model.hyperedge_contrastive_loss(ze, incidence)
            loss = ht_loss + cfg["train"]["lambda_cl"] * cl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print(f"pretrain_epoch={epoch} loss={total:.4f}")

    # Stage 2 initializes relation-aware reasoning from the pre-trained embeddings
    # and adds embedding-level plus score-level perturbation consistency.
    print("Stage 2: relation-aware reasoning with perturbation consistency")
    best_mrr = -1.0
    patience = 0
    best_state = None
    for epoch in range(1, cfg["train"]["finetune_epochs"] + 1):
        model.train()
        total = 0.0
        samples = dataset.negative_sample(dataset.train, cfg["train"]["negative_ratio"])
        for batch in tqdm(batch_iter(samples, cfg["train"]["batch_size"]), desc=f"finetune {epoch}", leave=False):
            h, r, t, y = dataset.as_tensors(batch, device)
            ze, zr = model.encode(incidence, text, visual)
            score, emb_loss, score_loss = model.perturbation_loss(
                ze,
                zr,
                h,
                r,
                t,
                cfg["train"]["sigma_entity"],
                cfg["train"]["sigma_relation"],
            )
            kgc_loss = F.binary_cross_entropy_with_logits(score, y)
            loss = kgc_loss + cfg["train"]["lambda_emb"] * emb_loss + cfg["train"]["lambda_score"] * score_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())

        print(f"finetune_epoch={epoch} loss={total:.4f}")
        if epoch % cfg["train"]["eval_every"] == 0:
            metrics = filtered_link_prediction(model, dataset, incidence, text, visual, dataset.valid, device)
            print(f"valid {metrics}")
            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                patience = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["train"]["early_stop_patience"]:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = filtered_link_prediction(model, dataset, incidence, text, visual, dataset.test, device)
    print(f"test {test_metrics}")


if __name__ == "__main__":
    main()
