import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    def __init__(self, dim, layers, dropout):
        super().__init__()
        self.weights = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, incidence):
        h = incidence
        dv = h.sum(dim=1).clamp_min(1.0).pow(-0.5)
        de = h.sum(dim=0).clamp_min(1.0).pow(-1.0)
        norm_left = dv[:, None] * h
        norm_right = h.t() * dv[None, :]
        for layer in self.weights:
            x = norm_left @ (de[:, None] * (norm_right @ x))
            x = F.relu(layer(x))
            x = self.dropout(x)
        return x


class TripleScorer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, zh, zr, zt):
        u = torch.cat([zh, zr, zt, zh * zr, zr * zt, zh * zt], dim=-1)
        return self.net(u).squeeze(-1)


class HERO(nn.Module):
    def __init__(self, num_entities, num_relations, text_dim, visual_dim, dim=256, hidden_dim=512, hyper_layers=2, dropout=0.2):
        super().__init__()
        self.entity_fallback = nn.Embedding(num_entities, dim)
        self.text_proj = nn.Linear(text_dim, dim) if text_dim else None
        self.visual_proj = nn.Linear(visual_dim, dim) if visual_dim else None
        self.fuse = nn.Linear(dim * (int(text_dim > 0) + int(visual_dim > 0) or 1), dim)
        self.hyper = HypergraphConv(dim, hyper_layers, dropout)
        self.relation_bias = nn.Embedding(num_relations, dim)
        self.pretrain_scorer = TripleScorer(dim, hidden_dim, dropout)
        self.reason_scorer = TripleScorer(dim, hidden_dim, dropout)
        self.global_gate = nn.Linear(dim * 3, dim)
        self.discriminator = nn.Bilinear(dim, dim, 1)

    def initial_entities(self, text_features=None, visual_features=None):
        parts = []
        if text_features is not None and self.text_proj is not None:
            parts.append(F.relu(self.text_proj(text_features)))
        if visual_features is not None and self.visual_proj is not None:
            parts.append(F.relu(self.visual_proj(visual_features)))
        if not parts:
            return self.entity_fallback.weight
        return F.relu(self.fuse(torch.cat(parts, dim=-1)))

    def encode(self, incidence, text_features=None, visual_features=None):
        x0 = self.initial_entities(text_features, visual_features)
        ze = self.hyper(x0, incidence)
        de = incidence.sum(dim=0).clamp_min(1.0)
        zr = (incidence.t() @ ze) / de[:, None]
        zr = zr + self.relation_bias.weight
        return ze, zr

    def pretrain_score(self, ze, zr, h, r, t):
        return self.pretrain_scorer(ze[h], zr[r], ze[t])

    def reason_score(self, ze, zr, h, r, t):
        zh, zt, zrel = ze[h], ze[t], zr[r]
        gate = torch.sigmoid(self.global_gate(torch.cat([zh, zrel, zt], dim=-1)))
        zh_ctx = gate * zh + (1.0 - gate) * zrel
        zt_ctx = gate * zt + (1.0 - gate) * zrel
        return self.reason_scorer(zh_ctx, zrel, zt_ctx), zh_ctx, zt_ctx

    def hyperedge_contrastive_loss(self, ze, incidence, replace_ratio=0.2):
        de = incidence.sum(dim=0).clamp_min(1.0)
        real = (incidence.t() @ ze) / de[:, None]
        global_vec = torch.tanh(real.mean(dim=0, keepdim=True)).expand_as(real)

        noise = incidence.clone()
        if replace_ratio > 0:
            mask = torch.rand_like(noise) < (replace_ratio * noise.mean().clamp_min(1e-6))
            noise = torch.where(mask, 1.0 - noise, noise)
        pert_de = noise.sum(dim=0).clamp_min(1.0)
        pert = (noise.t() @ ze) / pert_de[:, None]

        pos = self.discriminator(real, global_vec).squeeze(-1)
        neg = self.discriminator(pert, global_vec).squeeze(-1)
        ones = torch.ones_like(pos)
        zeros = torch.zeros_like(neg)
        return F.binary_cross_entropy_with_logits(pos, ones) + F.binary_cross_entropy_with_logits(neg, zeros)

    def perturbation_loss(self, ze, zr, h, r, t, sigma_entity, sigma_relation):
        clean_score, clean_h, clean_t = self.reason_score(ze, zr, h, r, t)
        ze_p = ze + torch.randn_like(ze) * sigma_entity
        zr_p = zr + torch.randn_like(zr) * sigma_relation
        pert_score, pert_h, pert_t = self.reason_score(ze_p, zr_p, h, r, t)
        emb_loss = F.mse_loss(pert_h, clean_h.detach()) + F.mse_loss(pert_t, clean_t.detach())
        score_loss = F.mse_loss(torch.sigmoid(pert_score), torch.sigmoid(clean_score.detach()))
        return clean_score, emb_loss, score_loss

