"""UserTower, AdTower, TwoTower, CrossTower and ThreeTower: the Pinterest-style co-train model
(notebook section 6).

TE_USER/TE_AD/TE_CROSS implement PROMPT_taobao_ctr_mlops.md constraint 4: user-only click rates
may enter the user tower, ad-only rates the ad tower, and the user x category rate only where
user and ad meet (the cross tower / inter module) -- it would break ad-tower caching otherwise.
Indices follow features.click_rates.TE_COLS ordering.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from taobao_ctr.models.layers import AD_FEATS, N_AD, N_USER, USER_FEATS, Embeddings, mlp

TE_USER, TE_AD, TE_CROSS = [0], [1, 2, 3, 4, 5, 7], [6]


class UserTower(nn.Module):
    """User side only: profile + context + averaged history + user click rate."""

    def __init__(self, emb: Embeddings, hp: dict):
        super().__init__()
        self.emb = emb
        self.net = mlp((N_USER + 1) * hp["emb_dim"] + len(TE_USER), [256, 128], hp["tower_dim"], hp["dropout"])

    def forward(self, b: dict) -> torch.Tensor:
        return self.net(torch.cat([self.emb.ids(b, USER_FEATS), self.emb.mean_history(b), b["te"][:, TE_USER]], -1))


class AdTower(nn.Module):
    """Ad side only (Pinterest's 'Pin tower'): can be computed offline and cached per ad."""

    def __init__(self, emb: Embeddings, hp: dict):
        super().__init__()
        self.emb = emb
        n = N_AD * hp["emb_dim"] + len(TE_AD)
        self.net = mlp(n, [256, 128], hp["tower_dim"], hp["dropout"])
        self.to_cross = nn.Linear(n, hp["tower_dim"])  # what the ad 'asks' the cross tower (also cacheable)

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.emb.ids(b, AD_FEATS), b["te"][:, TE_AD]], -1)
        return self.net(x), self.to_cross(x)


class TwoTower(nn.Module):
    def __init__(self, hp: dict, sizes: dict, seq_len: int | None = None):
        super().__init__()
        self.emb = Embeddings(hp["emb_dim"], sizes)
        self.user, self.ad = UserTower(self.emb, hp), AdTower(self.emb, hp)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, b: dict) -> dict:
        return {"logit": (self.user(b) * self.ad(b)[0]).sum(-1) + self.bias}


class CrossTower(nn.Module):
    """The ad's query tokens attend over the user's history: this is where user and ad meet."""

    def __init__(self, emb: Embeddings, hp: dict, seq_len: int):
        super().__init__()
        D, self.nq = hp["tower_dim"], hp["n_query_tokens"]
        self.emb = emb
        self.proj, self.pos, self.norm = nn.Linear(hp["emb_dim"], D), nn.Embedding(seq_len, D), nn.LayerNorm(D)
        self.queries = nn.Linear(D, self.nq * D)
        self.empty = nn.Parameter(torch.zeros(1, 1, D))  # always-present token (users with no history)
        self.attn = nn.MultiheadAttention(D, hp["n_heads"], batch_first=True)
        self.out = nn.Linear(D, D)

    def forward(self, ad_query: torch.Tensor, b: dict) -> torch.Tensor:
        B = ad_query.shape[0]
        k = self.norm(self.proj(self.emb.history(b)) + self.pos.weight)
        k = torch.cat([self.empty.expand(B, 1, -1), k], 1)
        pad = torch.cat([torch.zeros_like(b["hist_btag"][:, :1], dtype=torch.bool), b["hist_btag"] == 0], 1)
        q = self.queries(ad_query).view(B, self.nq, -1)
        a, _ = self.attn(q, k, k, key_padding_mask=pad)
        return self.out(a.mean(1))


class ThreeTower(nn.Module):
    """Two-tower + cross tower + inter module. Two heads: rich 3-tower score and cheap
    two-tower score (co-trained)."""

    def __init__(self, hp: dict, sizes: dict, seq_len: int):
        super().__init__()
        D = hp["tower_dim"]
        self.emb = Embeddings(hp["emb_dim"], sizes)
        self.user, self.ad, self.cross = UserTower(self.emb, hp), AdTower(self.emb, hp), CrossTower(self.emb, hp, seq_len)
        self.inter = mlp(3 * D + len(TE_CROSS), [128, 64], 1, hp["dropout"])
        self.task_bias = nn.Linear(2 * D, 1)
        self.dot_bias = nn.Parameter(torch.zeros(1))

    def forward(self, b: dict) -> dict:
        u = self.user(b)
        a, a_query = self.ad(b)
        c = self.cross(a_query, b)
        rich = self.inter(torch.cat([u, a, c, b["te"][:, TE_CROSS]], -1)) + self.task_bias(torch.cat([u, a], -1))
        return {"logit": rich.squeeze(-1), "cotrain_logit": (u * a).sum(-1) + self.dot_bias}
