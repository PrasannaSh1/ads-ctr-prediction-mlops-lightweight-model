"""Forward-shape smoke tests for every registered model (notebook section 6): correct logit
shape for a batch of B examples, and the 3-tower model must return both its rich head and its
cheap co-trained two-tower head (PROMPT_taobao_ctr_mlops.md constraint 5)."""
from __future__ import annotations

import torch

from taobao_ctr.data.impressions import CTX_COLS, ID_COLS, USER_COLS
from taobao_ctr.models.registry import MODELS

SEQ_LEN = 8
B = 6
TABLE_NAMES = (
    set(USER_COLS) | set(CTX_COLS)
    | {"adgroup_id", "campaign_id", "customer", "cate", "brand", "price_bucket", "btag"}
)
SIZES = {k: 10 for k in TABLE_NAMES}
HP = {"emb_dim": 4, "tower_dim": 8, "n_heads": 2, "n_query_tokens": 2, "dropout": 0.0}


def _fake_batch() -> dict:
    b = {c: torch.randint(0, 10, (B,)) for c in ID_COLS}
    b["hist_cate"] = torch.randint(0, 10, (B, SEQ_LEN))
    b["hist_brand"] = torch.randint(0, 10, (B, SEQ_LEN))
    b["hist_btag"] = torch.randint(0, 10, (B, SEQ_LEN))
    b["te"] = torch.randn(B, 8)
    b["label"] = torch.randint(0, 2, (B,)).float()
    return b


def test_all_models_forward_shapes():
    batch = _fake_batch()
    for name, cls in MODELS.items():
        model = cls(HP, SIZES, SEQ_LEN)
        out = model(batch)
        assert out["logit"].shape == (B,), f"{name}: unexpected logit shape {out['logit'].shape}"


def test_three_tower_returns_both_heads():
    batch = _fake_batch()
    model = MODELS["3-tower co-train"](HP, SIZES, SEQ_LEN)
    out = model(batch)
    assert set(out) == {"logit", "cotrain_logit"}
    assert out["cotrain_logit"].shape == (B,)


def test_only_three_tower_has_a_second_head():
    batch = _fake_batch()
    for name, cls in MODELS.items():
        if name == "3-tower co-train":
            continue
        out = cls(HP, SIZES, SEQ_LEN)(batch)
        assert set(out) == {"logit"}, f"{name}: unexpectedly has extra heads {set(out) - {'logit'}}"
