"""Checks that every model runs on a small fake batch and returns the right shapes,
and that only the 3-tower model returns a second (cheap) head."""
import torch

from taobao_ctr.data.impressions import CTX_COLS, ID_COLS, USER_COLS
from taobao_ctr.models.registry import MODELS

SEQ_LEN = 8
BATCH = 6

# Every embedding table gets 10 possible values in this test
TABLE_NAMES = USER_COLS + CTX_COLS + ["adgroup_id", "campaign_id", "customer", "cate", "brand", "price_bucket", "btag"]
SIZES = {}
for name in TABLE_NAMES:
    SIZES[name] = 10

HP = {"emb_dim": 4, "tower_dim": 8, "n_heads": 2, "n_query_tokens": 2, "dropout": 0.0}


def fake_batch():
    batch = {}
    for col in ID_COLS:
        batch[col] = torch.randint(0, 10, (BATCH,))
    batch["hist_cate"] = torch.randint(0, 10, (BATCH, SEQ_LEN))
    batch["hist_brand"] = torch.randint(0, 10, (BATCH, SEQ_LEN))
    batch["hist_btag"] = torch.randint(0, 10, (BATCH, SEQ_LEN))
    batch["te"] = torch.randn(BATCH, 8)
    batch["label"] = torch.randint(0, 2, (BATCH,)).float()
    return batch


def test_all_models_forward_shapes():
    batch = fake_batch()
    for name, model_class in MODELS.items():
        model = model_class(HP, SIZES, SEQ_LEN)
        output = model(batch)
        assert output["logit"].shape == (BATCH,), f"{name}: unexpected logit shape {output['logit'].shape}"


def test_three_tower_returns_both_heads():
    batch = fake_batch()
    model = MODELS["3-tower co-train"](HP, SIZES, SEQ_LEN)
    output = model(batch)
    assert set(output.keys()) == {"logit", "cotrain_logit"}
    assert output["cotrain_logit"].shape == (BATCH,)


def test_only_three_tower_has_a_second_head():
    batch = fake_batch()
    for name, model_class in MODELS.items():
        if name == "3-tower co-train":
            continue
        model = model_class(HP, SIZES, SEQ_LEN)
        output = model(batch)
        assert set(output.keys()) == {"logit"}, f"{name}: has extra heads {set(output.keys()) - {'logit'}}"
