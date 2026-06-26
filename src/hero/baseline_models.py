"""Reference baseline model implementations for MMKGC comparison experiments.

The classes in this file provide a unified, lightweight source-code scaffold for
the baseline families discussed in the manuscript. They are intentionally small
and dependency-free beyond PyTorch. They should be treated as reference
implementations and extension points, not as official reproductions of the
corresponding papers.
"""

import math
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseKGCModel(nn.Module):
    """Common embedding table and scoring API for KGC/MMKGC baselines."""

    def __init__(self, num_entities, num_relations, dim=256, margin=6.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin = margin
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embeddings with the scale commonly used by KGE models."""
        bound = 6.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        """Return entity embeddings; multimodal subclasses may override this."""
        return self.entity(entity_ids)

    def relation_repr(self, relation_ids):
        """Return relation embeddings for the queried relation IDs."""
        return self.relation(relation_ids)

    def score(self, h, r, t, text_features=None, visual_features=None):
        """Compute larger-is-better plausibility scores for triples."""
        zh = self.entity_repr(h, text_features, visual_features)
        zr = self.relation_repr(r)
        zt = self.entity_repr(t, text_features, visual_features)
        return self._score_embeddings(zh, zr, zt)

    def forward(self, h, r, t, text_features=None, visual_features=None):
        """Alias score() so the classes can be used as normal nn.Modules."""
        return self.score(h, r, t, text_features, visual_features)

    def _score_embeddings(self, zh, zr, zt):
        raise NotImplementedError


class TransE(BaseKGCModel):
    """Translation baseline: h + r should be close to t."""

    def _score_embeddings(self, zh, zr, zt):
        return self.margin - torch.linalg.vector_norm(zh + zr - zt, ord=1, dim=-1)


class DistMult(BaseKGCModel):
    """Diagonal bilinear semantic matching baseline."""

    def _score_embeddings(self, zh, zr, zt):
        return torch.sum(zh * zr * zt, dim=-1)


class ComplEx(BaseKGCModel):
    """Complex-valued bilinear baseline implemented with split real/imag parts."""

    def __init__(self, num_entities, num_relations, dim=256, margin=6.0):
        if dim % 2 != 0:
            raise ValueError("ComplEx requires an even embedding dimension.")
        super().__init__(num_entities, num_relations, dim, margin)

    def _score_embeddings(self, zh, zr, zt):
        h_re, h_im = zh.chunk(2, dim=-1)
        r_re, r_im = zr.chunk(2, dim=-1)
        t_re, t_im = zt.chunk(2, dim=-1)
        return torch.sum(
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re,
            dim=-1,
        )


class RotatE(BaseKGCModel):
    """Rotation baseline where relations rotate complex entity embeddings."""

    def __init__(self, num_entities, num_relations, dim=256, margin=6.0):
        if dim % 2 != 0:
            raise ValueError("RotatE requires an even embedding dimension.")
        super().__init__(num_entities, num_relations, dim, margin)

    def _score_embeddings(self, zh, zr, zt):
        h_re, h_im = zh.chunk(2, dim=-1)
        t_re, t_im = zt.chunk(2, dim=-1)
        phase = zr[..., : self.dim // 2] / (self.margin / math.pi)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re
        return self.margin - torch.linalg.vector_norm(torch.cat([rot_re - t_re, rot_im - t_im], dim=-1), dim=-1)


class QuatE(BaseKGCModel):
    """Quaternion interaction baseline with Hamilton product scoring."""

    def __init__(self, num_entities, num_relations, dim=256, margin=6.0):
        if dim % 4 != 0:
            raise ValueError("QuatE requires a dimension divisible by 4.")
        super().__init__(num_entities, num_relations, dim, margin)

    def _score_embeddings(self, zh, zr, zt):
        h0, h1, h2, h3 = zh.chunk(4, dim=-1)
        r0, r1, r2, r3 = F.normalize(zr, p=2, dim=-1).chunk(4, dim=-1)
        t0, t1, t2, t3 = zt.chunk(4, dim=-1)
        p0 = h0 * r0 - h1 * r1 - h2 * r2 - h3 * r3
        p1 = h0 * r1 + h1 * r0 + h2 * r3 - h3 * r2
        p2 = h0 * r2 - h1 * r3 + h2 * r0 + h3 * r1
        p3 = h0 * r3 + h1 * r2 - h2 * r1 + h3 * r0
        return torch.sum(p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3, dim=-1)


class DualE(QuatE):
    """Dual-quaternion-inspired baseline using the QuatE scaffold as a compact proxy."""


class BiQUE(QuatE):
    """Bidirectional quaternion-style baseline with a symmetric reverse score."""

    def _score_embeddings(self, zh, zr, zt):
        forward = super()._score_embeddings(zh, zr, zt)
        reverse = super()._score_embeddings(zt, zr, zh)
        return 0.5 * (forward + reverse)


class MultimodalFusionModel(BaseKGCModel):
    """Shared text/image projection block for multimodal baseline scaffolds."""

    def __init__(self, num_entities, num_relations, text_dim=0, visual_dim=0, dim=256, margin=6.0, dropout=0.2):
        super().__init__(num_entities, num_relations, dim, margin)
        self.text_proj = nn.Linear(text_dim, dim) if text_dim else None
        self.visual_proj = nn.Linear(visual_dim, dim) if visual_dim else None
        self.dropout = nn.Dropout(dropout)
        part_count = 1 + int(text_dim > 0) + int(visual_dim > 0)
        self.fusion = nn.Linear(part_count * dim, dim)

    def _project_modalities(self, entity_ids, text_features=None, visual_features=None):
        parts = [self.entity(entity_ids)]
        if text_features is not None and self.text_proj is not None:
            parts.append(F.relu(self.text_proj(text_features[entity_ids])))
        if visual_features is not None and self.visual_proj is not None:
            parts.append(F.relu(self.visual_proj(visual_features[entity_ids])))
        return parts

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        parts = self._project_modalities(entity_ids, text_features, visual_features)
        return self.dropout(F.relu(self.fusion(torch.cat(parts, dim=-1))))

    def _score_embeddings(self, zh, zr, zt):
        return torch.sum(zh * zr * zt, dim=-1)


class IKRL(MultimodalFusionModel):
    """Image-enhanced KRL scaffold with visual features favored when available."""

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        if visual_features is None or self.visual_proj is None:
            return super().entity_repr(entity_ids, text_features, visual_features)
        base = self.entity(entity_ids)
        visual = F.relu(self.visual_proj(visual_features[entity_ids]))
        return 0.5 * (base + visual)


class TransAE(MultimodalFusionModel):
    """Autoencoder-style fusion baseline with a compact reconstruction head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = nn.Linear(self.dim, self.dim)

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        fused = super().entity_repr(entity_ids, text_features, visual_features)
        return 0.5 * (fused + self.decoder(fused))


class VBKGC(MultimodalFusionModel):
    """Variational-style multimodal scaffold using stochastic entity states."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logvar = nn.Linear(self.dim, self.dim)

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        mu = super().entity_repr(entity_ids, text_features, visual_features)
        if not self.training:
            return mu
        std = torch.exp(0.5 * self.logvar(mu)).clamp_max(2.0)
        return mu + torch.randn_like(std) * std


class OTKGE(MultimodalFusionModel):
    """Optimal-transport-inspired scaffold using modality alignment regularization."""

    def alignment_loss(self, entity_ids, text_features=None, visual_features=None):
        parts = self._project_modalities(entity_ids, text_features, visual_features)
        if len(parts) < 2:
            return torch.zeros((), device=self.entity.weight.device)
        anchor = parts[0].detach()
        return sum(F.mse_loss(part, anchor) for part in parts[1:]) / (len(parts) - 1)


class QEB(MultimodalFusionModel):
    """Query-enhanced baseline that gates entity features by relation context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_gate = nn.Linear(self.dim * 2, self.dim)

    def score(self, h, r, t, text_features=None, visual_features=None):
        zh = super().entity_repr(h, text_features, visual_features)
        zr = self.relation_repr(r)
        zt = super().entity_repr(t, text_features, visual_features)
        gate_h = torch.sigmoid(self.query_gate(torch.cat([zh, zr], dim=-1)))
        gate_t = torch.sigmoid(self.query_gate(torch.cat([zt, zr], dim=-1)))
        return self._score_embeddings(gate_h * zh, zr, gate_t * zt)


class VISTA(MultimodalFusionModel):
    """Transformer-like visual-text scaffold with a lightweight attention layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = nn.MultiheadAttention(self.dim, num_heads=4, batch_first=True)

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        parts = self._project_modalities(entity_ids, text_features, visual_features)
        tokens = torch.stack(parts, dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return attended.mean(dim=1)


class AdaMF(MultimodalFusionModel):
    """Adaptive multimodal fusion baseline with learned per-entity modality weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_head = nn.Linear(self.dim, 1)

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        parts = self._project_modalities(entity_ids, text_features, visual_features)
        stacked = torch.stack(parts, dim=1)
        weights = torch.softmax(self.weight_head(stacked).squeeze(-1), dim=-1)
        return torch.sum(stacked * weights.unsqueeze(-1), dim=1)


class MyGO(MultimodalFusionModel):
    """Token-style multimodal scaffold with a small feed-forward token mixer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixer = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        return self.mixer(super().entity_repr(entity_ids, text_features, visual_features))


class KON(MultimodalFusionModel):
    """Knowledge-oriented multimodal scaffold with relation-aware normalization."""

    def score(self, h, r, t, text_features=None, visual_features=None):
        zh = F.normalize(self.entity_repr(h, text_features, visual_features), dim=-1)
        zr = F.normalize(self.relation_repr(r), dim=-1)
        zt = F.normalize(self.entity_repr(t, text_features, visual_features), dim=-1)
        return self._score_embeddings(zh, zr, zt)


class MoSE(MultimodalFusionModel):
    """Ensemble baseline that averages structural, textual, and visual scores."""

    def score(self, h, r, t, text_features=None, visual_features=None):
        zr = self.relation_repr(r)
        scores = [self._score_embeddings(self.entity(h), zr, self.entity(t))]
        if text_features is not None and self.text_proj is not None:
            text = F.relu(self.text_proj(text_features))
            scores.append(self._score_embeddings(text[h], zr, text[t]))
        if visual_features is not None and self.visual_proj is not None:
            visual = F.relu(self.visual_proj(visual_features))
            scores.append(self._score_embeddings(visual[h], zr, visual[t]))
        return torch.stack(scores, dim=0).mean(dim=0)


class MMRNS(MultimodalFusionModel):
    """MMRNS-style scaffold with relation-conditioned hard negative utilities."""

    def relation_conditioned_score(self, h, r, candidates, text_features=None, visual_features=None):
        r_batch = r.expand_as(candidates)
        h_batch = h.expand_as(candidates)
        return self.score(h_batch, r_batch, candidates, text_features, visual_features)


class IMF(MultimodalFusionModel):
    """Interactive multimodal fusion scaffold using pairwise modality products."""

    def entity_repr(self, entity_ids, text_features=None, visual_features=None):
        parts = self._project_modalities(entity_ids, text_features, visual_features)
        fused = torch.cat(parts, dim=-1)
        base = F.relu(self.fusion(fused))
        if len(parts) >= 2:
            base = base + sum(parts[i] * parts[j] for i in range(len(parts)) for j in range(i + 1, len(parts))) / len(parts)
        return base


class MoMoK(MoSE):
    """Modality-preserving ensemble scaffold with a learnable score combiner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_weights = nn.Parameter(torch.ones(3))

    def score(self, h, r, t, text_features=None, visual_features=None):
        zr = self.relation_repr(r)
        score_parts = [self._score_embeddings(self.entity(h), zr, self.entity(t))]
        if text_features is not None and self.text_proj is not None:
            text = F.relu(self.text_proj(text_features))
            score_parts.append(self._score_embeddings(text[h], zr, text[t]))
        if visual_features is not None and self.visual_proj is not None:
            visual = F.relu(self.visual_proj(visual_features))
            score_parts.append(self._score_embeddings(visual[h], zr, visual[t]))
        weights = torch.softmax(self.score_weights[: len(score_parts)], dim=0)
        return torch.sum(torch.stack(score_parts, dim=0) * weights[:, None], dim=0)


class MHyper(MultimodalFusionModel):
    """Hypercomplex multimodal scaffold represented by a quaternion scorer."""

    def __init__(self, num_entities, num_relations, text_dim=0, visual_dim=0, dim=256, margin=6.0, dropout=0.2):
        if dim % 4 != 0:
            raise ValueError("MHyper requires a dimension divisible by 4.")
        super().__init__(num_entities, num_relations, text_dim, visual_dim, dim, margin, dropout)
        self.quat = QuatE(num_entities, num_relations, dim, margin)

    def _score_embeddings(self, zh, zr, zt):
        return self.quat._score_embeddings(zh, zr, zt)


BASELINE_REGISTRY: Dict[str, Type[BaseKGCModel]] = {
    "TransE": TransE,
    "DistMult": DistMult,
    "ComplEx": ComplEx,
    "RotatE": RotatE,
    "QuatE": QuatE,
    "DualE": DualE,
    "BiQUE": BiQUE,
    "IKRL": IKRL,
    "TransAE": TransAE,
    "VBKGC": VBKGC,
    "OTKGE": OTKGE,
    "QEB": QEB,
    "VISTA": VISTA,
    "AdaMF": AdaMF,
    "MyGO": MyGO,
    "K-ON": KON,
    "MoSE": MoSE,
    "MMRNS": MMRNS,
    "IMF": IMF,
    "MoMoK": MoMoK,
    "M-Hyper": MHyper,
}


def build_baseline(name, *args, **kwargs):
    """Instantiate a baseline by manuscript/table name."""
    if name not in BASELINE_REGISTRY:
        options = ", ".join(sorted(BASELINE_REGISTRY))
        raise KeyError(f"Unknown baseline {name!r}. Available baselines: {options}")
    return BASELINE_REGISTRY[name](*args, **kwargs)
