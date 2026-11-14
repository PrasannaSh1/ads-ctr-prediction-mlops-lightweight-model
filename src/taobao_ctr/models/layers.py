"""Building blocks shared by every deep model: a simple MLP, and the embedding tables for the
id columns and the behaviour history."""
import torch
import torch.nn as nn

from taobao_ctr.data.impressions import CTX_COLS, USER_COLS

# (name of the column in a batch, name of the embedding table to use)
USER_FEATS = []
for col in USER_COLS + CTX_COLS:
    USER_FEATS.append((col, col))

AD_FEATS = [
    ("adgroup_id", "adgroup_id"),
    ("campaign_id", "campaign_id"),
    ("customer", "customer"),
    ("ad_cate", "cate"),      # the ad's category uses the "cate" table, shared with the history
    ("ad_brand", "brand"),    # the ad's brand uses the "brand" table, shared with the history
    ("price_bucket", "price_bucket"),
]
N_USER = len(USER_FEATS)  # 10
N_AD = len(AD_FEATS)  # 6


def mlp(n_in, hidden_sizes, n_out, dropout):
    """A stack of Linear -> ReLU -> Dropout layers, ending with one Linear layer."""
    layers = []
    for size in hidden_sizes:
        layers.append(nn.Linear(n_in, size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        n_in = size
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class Embeddings(nn.Module):
    """One embedding table per id column. Row 0 of every table is the padding row (all zeros)."""

    def __init__(self, dim, sizes):
        super().__init__()
        tables = {}
        for name, n_values in sizes.items():
            tables[name] = nn.Embedding(n_values + 1, dim, padding_idx=0)
        # self.t holds the tables. (The short name is kept so saved models still load.)
        self.t = nn.ModuleDict(tables)

    def ids(self, batch, feats):
        """Look up several id columns and join their vectors side by side."""
        vectors = []
        for batch_key, table_name in feats:
            vectors.append(self.t[table_name](batch[batch_key]))
        return torch.cat(vectors, -1)

    def history(self, batch):
        """One vector per past action: category + brand + action type. Shape (B, seq_len, dim)."""
        return self.t["cate"](batch["hist_cate"]) + self.t["brand"](batch["hist_brand"]) + self.t["btag"](batch["hist_btag"])

    def mean_history(self, batch):
        """The average of the real (non padding) past actions. Shape (B, dim)."""
        h = self.history(batch)
        mask = (batch["hist_btag"] > 0).float().unsqueeze(-1)  # 1 for a real action, 0 for padding
        total = (h * mask).sum(1)
        count = mask.sum(1).clamp(min=1)  # at least 1, so a user with no history does not divide by 0
        return total / count
