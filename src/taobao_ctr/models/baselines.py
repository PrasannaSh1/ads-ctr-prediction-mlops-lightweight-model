"""The MLP, MLP + sequence and Transformer (BST style) models.

Every model is created the same way: ModelClass(hp, sizes, seq_len). Only the Transformer
really needs seq_len (for its position embeddings).
"""
import torch
import torch.nn as nn

from taobao_ctr.models.layers import AD_FEATS, N_AD, N_USER, USER_FEATS, Embeddings, mlp


class MLP(nn.Module):
    """All id embeddings + the 8 click rates, through an MLP."""

    def __init__(self, hp, sizes, seq_len=None):
        super().__init__()
        d = hp["emb_dim"]
        self.emb = Embeddings(d, sizes)
        n_in = (N_USER + N_AD) * d + 8  # 16 ids x 16 numbers + 8 click rates = 264
        self.net = mlp(n_in, [256, 128, 64], 1, hp["dropout"])

    def forward(self, batch):
        x = torch.cat([self.emb.ids(batch, USER_FEATS), self.emb.ids(batch, AD_FEATS), batch["te"]], -1)
        logit = self.net(x).squeeze(-1)
        return {"logit": logit}


class MLPSequence(nn.Module):
    """Same as MLP, plus the average of the user's past actions."""

    def __init__(self, hp, sizes, seq_len=None):
        super().__init__()
        d = hp["emb_dim"]
        self.emb = Embeddings(d, sizes)
        n_in = (N_USER + N_AD + 1) * d + 8  # one extra vector for the history
        self.net = mlp(n_in, [256, 128, 64], 1, hp["dropout"])

    def forward(self, batch):
        x = torch.cat([
            self.emb.ids(batch, USER_FEATS),
            self.emb.ids(batch, AD_FEATS),
            self.emb.mean_history(batch),
            batch["te"],
        ], -1)
        logit = self.net(x).squeeze(-1)
        return {"logit": logit}


class Transformer(nn.Module):
    """The 50 past actions plus the target ad as the last token, through self-attention."""

    def __init__(self, hp, sizes, seq_len=None):
        super().__init__()
        d = hp["emb_dim"]
        D = hp["tower_dim"]
        self.emb = Embeddings(d, sizes)
        self.tok = nn.Linear(d, D)  # lift each token from d to D numbers
        self.pos = nn.Embedding(seq_len + 1, D)  # a learned vector per position (50 actions + the ad)
        layer = nn.TransformerEncoderLayer(
            D, hp["n_heads"], 2 * D, dropout=hp["dropout"], batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        n_in = (N_USER + N_AD) * d + 2 * D + 8
        self.net = mlp(n_in, [256, 128, 64], 1, hp["dropout"])

    def forward(self, batch):
        # The ad, built like an action (category + brand), so it lives in the same space
        target = self.emb.t["cate"](batch["ad_cate"]) + self.emb.t["brand"](batch["ad_brand"])
        # 50 history tokens + 1 ad token, plus the position vectors
        tokens = torch.cat([self.emb.history(batch), target.unsqueeze(1)], 1)
        x = self.tok(tokens) + self.pos.weight
        # Padding slots are ignored by the attention; the ad token never is
        is_padding = batch["hist_btag"] == 0
        ad_not_padding = torch.zeros_like(batch["hist_btag"][:, :1], dtype=torch.bool)
        pad = torch.cat([is_padding, ad_not_padding], 1)
        z = self.enc(x, src_key_padding_mask=pad)
        # Average of the real outputs
        keep = (~pad).float().unsqueeze(-1)
        mean = (z * keep).sum(1) / keep.sum(1)
        x = torch.cat([
            self.emb.ids(batch, USER_FEATS),
            self.emb.ids(batch, AD_FEATS),
            z[:, -1],  # the ad token's output, after it has looked at the history
            mean,
            batch["te"],
        ], -1)
        logit = self.net(x).squeeze(-1)
        return {"logit": logit}
