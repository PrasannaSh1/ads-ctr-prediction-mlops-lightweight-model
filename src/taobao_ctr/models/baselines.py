"""MLP, MLPSequence and Transformer (BST-style) models (notebook section 6).

Every model shares the uniform constructor signature `(hp, sizes, seq_len=None)` so
training/tuning code can instantiate any registered model the same way; only Transformer
actually uses seq_len (for its positional embedding table).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from taobao_ctr.models.layers import AD_FEATS, N_AD, N_USER, USER_FEATS, Embeddings, mlp


class MLP(nn.Module):
    def __init__(self, hp: dict, sizes: dict, seq_len: int | None = None):
        super().__init__()
        d = hp["emb_dim"]
        self.emb = Embeddings(d, sizes)
        self.net = mlp((N_USER + N_AD) * d + 8, [256, 128, 64], 1, hp["dropout"])

    def forward(self, b: dict) -> dict:
        x = torch.cat([self.emb.ids(b, USER_FEATS), self.emb.ids(b, AD_FEATS), b["te"]], -1)
        return {"logit": self.net(x).squeeze(-1)}


class MLPSequence(nn.Module):
    def __init__(self, hp: dict, sizes: dict, seq_len: int | None = None):
        super().__init__()
        d = hp["emb_dim"]
        self.emb = Embeddings(d, sizes)
        self.net = mlp((N_USER + N_AD + 1) * d + 8, [256, 128, 64], 1, hp["dropout"])

    def forward(self, b: dict) -> dict:
        x = torch.cat([self.emb.ids(b, USER_FEATS), self.emb.ids(b, AD_FEATS), self.emb.mean_history(b), b["te"]], -1)
        return {"logit": self.net(x).squeeze(-1)}


class Transformer(nn.Module):
    """History tokens + the target ad as the last token, through self-attention (BST-style)."""

    def __init__(self, hp: dict, sizes: dict, seq_len: int | None = None):
        super().__init__()
        d, D = hp["emb_dim"], hp["tower_dim"]
        self.emb = Embeddings(d, sizes)
        self.tok = nn.Linear(d, D)
        self.pos = nn.Embedding(seq_len + 1, D)
        layer = nn.TransformerEncoderLayer(
            D, hp["n_heads"], 2 * D, dropout=hp["dropout"], batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.net = mlp((N_USER + N_AD) * d + 2 * D + 8, [256, 128, 64], 1, hp["dropout"])

    def forward(self, b: dict) -> dict:
        target = self.emb.t["cate"](b["ad_cate"]) + self.emb.t["brand"](b["ad_brand"])  # ad in the same space as actions
        x = self.tok(torch.cat([self.emb.history(b), target.unsqueeze(1)], 1)) + self.pos.weight
        pad = torch.cat([b["hist_btag"] == 0, torch.zeros_like(b["hist_btag"][:, :1], dtype=torch.bool)], 1)
        z = self.enc(x, src_key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        mean = (z * keep).sum(1) / keep.sum(1)
        x = torch.cat([self.emb.ids(b, USER_FEATS), self.emb.ids(b, AD_FEATS), z[:, -1], mean, b["te"]], -1)
        return {"logit": self.net(x).squeeze(-1)}
