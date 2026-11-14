"""The tower models: UserTower, AdTower, TwoTower, CrossTower and ThreeTower.

Where each click-rate feature goes (column numbers follow TE_COLS in features/click_rates.py):
  TE_USER  = the user's own click rate            -> user tower
  TE_AD    = the ad-only click rates + log price  -> ad tower
  TE_CROSS = the user x category click rate       -> only where user and ad meet (3-tower inter module)
If the user x category rate went into the ad tower, the ad vector would change per user and could
not be cached any more.
"""
import torch
import torch.nn as nn

from taobao_ctr.models.layers import AD_FEATS, N_AD, N_USER, USER_FEATS, Embeddings, mlp

TE_USER = [0]
TE_AD = [1, 2, 3, 4, 5, 7]
TE_CROSS = [6]


class UserTower(nn.Module):
    """User side only: profile + context ids, averaged history and the user's click rate."""

    def __init__(self, emb, hp):
        super().__init__()
        self.emb = emb
        n_in = (N_USER + 1) * hp["emb_dim"] + len(TE_USER)
        self.net = mlp(n_in, [256, 128], hp["tower_dim"], hp["dropout"])

    def forward(self, batch):
        x = torch.cat([
            self.emb.ids(batch, USER_FEATS),
            self.emb.mean_history(batch),
            batch["te"][:, TE_USER],
        ], -1)
        return self.net(x)


class AdTower(nn.Module):
    """Ad side only. It depends on nothing about the user, so it can be computed once per ad and cached."""

    def __init__(self, emb, hp):
        super().__init__()
        self.emb = emb
        n_in = N_AD * hp["emb_dim"] + len(TE_AD)
        self.net = mlp(n_in, [256, 128], hp["tower_dim"], hp["dropout"])
        self.to_cross = nn.Linear(n_in, hp["tower_dim"])  # the question the ad asks the cross tower (also cacheable)

    def forward(self, batch):
        x = torch.cat([self.emb.ids(batch, AD_FEATS), batch["te"][:, TE_AD]], -1)
        ad_vector = self.net(x)
        cross_query = self.to_cross(x)
        return ad_vector, cross_query


class TwoTower(nn.Module):
    """Score = user vector . ad vector + bias."""

    def __init__(self, hp, sizes, seq_len=None):
        super().__init__()
        self.emb = Embeddings(hp["emb_dim"], sizes)
        self.user = UserTower(self.emb, hp)
        self.ad = AdTower(self.emb, hp)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        u = self.user(batch)
        a, cross_query = self.ad(batch)
        logit = (u * a).sum(-1) + self.bias
        return {"logit": logit}


class CrossTower(nn.Module):
    """The ad asks questions about the user's history, using attention. This is where user and ad meet."""

    def __init__(self, emb, hp, seq_len):
        super().__init__()
        D = hp["tower_dim"]
        self.nq = hp["n_query_tokens"]  # how many questions the ad asks (4)
        self.emb = emb
        self.proj = nn.Linear(hp["emb_dim"], D)  # lift each action from emb_dim to D numbers
        self.pos = nn.Embedding(seq_len, D)  # a learned vector per position in the history
        self.norm = nn.LayerNorm(D)
        self.queries = nn.Linear(D, self.nq * D)  # one ad query -> nq query tokens
        self.empty = nn.Parameter(torch.zeros(1, 1, D))  # a token that is always there, for users with no history
        self.attn = nn.MultiheadAttention(D, hp["n_heads"], batch_first=True)
        self.out = nn.Linear(D, D)

    def forward(self, ad_query, batch):
        B = ad_query.shape[0]
        # Keys and values: the history actions, plus the "empty" token in front
        k = self.norm(self.proj(self.emb.history(batch)) + self.pos.weight)
        k = torch.cat([self.empty.expand(B, 1, -1), k], 1)
        # Mask: padding slots are ignored, the empty token never is
        empty_not_padding = torch.zeros_like(batch["hist_btag"][:, :1], dtype=torch.bool)
        pad = torch.cat([empty_not_padding, batch["hist_btag"] == 0], 1)
        # Queries: the ad's nq question tokens
        q = self.queries(ad_query).view(B, self.nq, -1)
        answers, weights = self.attn(q, k, k, key_padding_mask=pad)
        return self.out(answers.mean(1))  # average the nq answers into one vector


class ThreeTower(nn.Module):
    """Two-tower + cross tower + inter module. Returns two scores:
    "logit" (rich, from the inter module) and "cotrain_logit" (cheap, the two-tower dot product)."""

    def __init__(self, hp, sizes, seq_len):
        super().__init__()
        D = hp["tower_dim"]
        self.emb = Embeddings(hp["emb_dim"], sizes)
        self.user = UserTower(self.emb, hp)
        self.ad = AdTower(self.emb, hp)
        self.cross = CrossTower(self.emb, hp, seq_len)
        self.inter = mlp(3 * D + len(TE_CROSS), [128, 64], 1, hp["dropout"])
        self.task_bias = nn.Linear(2 * D, 1)
        self.dot_bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        u = self.user(batch)
        a, a_query = self.ad(batch)
        c = self.cross(a_query, batch)
        # Rich score: an MLP over user, ad, cross vector and the user x category click rate
        rich = self.inter(torch.cat([u, a, c, batch["te"][:, TE_CROSS]], -1))
        rich = rich + self.task_bias(torch.cat([u, a], -1))
        # Cheap score: the same dot product a two-tower model uses
        cheap = (u * a).sum(-1) + self.dot_bias
        return {"logit": rich.squeeze(-1), "cotrain_logit": cheap}
