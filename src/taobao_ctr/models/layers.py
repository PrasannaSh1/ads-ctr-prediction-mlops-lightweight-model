"""Shared building blocks for every deep model: a plain MLP and the embedding tables for ID
features and behaviour history (notebook section 6). All deep models share the same embedding
tables and click-rate features (see PROMPT_taobao_ctr_mlops.md constraint 4 for how the
click-rate columns are split by side, enforced in models/towers.py)."""
from __future__ import annotations

import torch
import torch.nn as nn

from taobao_ctr.data.impressions import CTX_COLS, USER_COLS

USER_FEATS = [(c, c) for c in USER_COLS + CTX_COLS]  # (batch key, embedding table)
AD_FEATS = [
    ("adgroup_id", "adgroup_id"), ("campaign_id", "campaign_id"), ("customer", "customer"),
    ("ad_cate", "cate"), ("ad_brand", "brand"), ("price_bucket", "price_bucket"),
]
N_USER, N_AD = len(USER_FEATS), len(AD_FEATS)


def mlp(n_in: int, hidden: list[int], n_out: int, dropout: float) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(dropout)]
        n_in = h
    return nn.Sequential(*layers, nn.Linear(n_in, n_out))


class Embeddings(nn.Module):
    """One table per ID feature; category and brand tables are shared by the ad and the history."""

    def __init__(self, d: int, sizes: dict):
        super().__init__()
        self.t = nn.ModuleDict({k: nn.Embedding(n + 1, d, padding_idx=0) for k, n in sizes.items()})

    def ids(self, b: dict, feats: list[tuple[str, str]]) -> torch.Tensor:
        return torch.cat([self.t[table](b[key]) for key, table in feats], -1)

    def history(self, b: dict) -> torch.Tensor:  # (B, L, d): one vector per past action
        return self.t["cate"](b["hist_cate"]) + self.t["brand"](b["hist_brand"]) + self.t["btag"](b["hist_btag"])

    def mean_history(self, b: dict) -> torch.Tensor:  # average of the real (non-padding) actions
        h, m = self.history(b), (b["hist_btag"] > 0).float().unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
