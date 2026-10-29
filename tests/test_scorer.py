"""TwoStageScorer: candidates in the cascade's top fraction must get exactly the full 3-tower's
own score; candidates outside it must get exactly the plain (cheap) two-tower score. Both are
checked against ThreeTower.forward() run directly end to end -- not against the scorer's own
internal AdTower/CrossTower calls -- so this is a genuine structural-equivalence check, not a
test of the same code path twice.

The user x category click-rate (TE_CROSS) is a documented serving-time simplification (always
0.0, see serving/scorer.py's docstring); the brute-force comparison here uses the same 0.0 so it
checks "does the scorer correctly decompose the model", not "does the simplification lose
accuracy" (a separate, known trade-off).
"""
from __future__ import annotations

import numpy as np
import torch

from taobao_ctr.data.dataset import build_dataset
from taobao_ctr.data.history import history
from taobao_ctr.models.layers import AD_FEATS, USER_FEATS
from taobao_ctr.models.towers import TE_AD, TE_CROSS, TE_USER, ThreeTower
from taobao_ctr.serving.ad_cache import _last_row_per_ad, build_ad_cache
from taobao_ctr.serving.scorer import TwoStageScorer, UserFeatureCache
from taobao_ctr.training.trainer import get_device

HP = {"emb_dim": 4, "tower_dim": 8, "n_heads": 2, "n_query_tokens": 2, "dropout": 0.0}
CASCADE_TOP_FRAC = 0.3


def _build_dataset(tmp_path, synthetic_params):
    p = synthetic_params
    dataset, _train, fit_rows, _val, _test = build_dataset(
        data_dir=tmp_path, kaggle_dataset="unused/unused", n_buckets=p.data.n_buckets,
        chunk_rows=p.data.chunk_rows, seq_len=p.data.seq_len, te_alpha=p.features.te_alpha,
    )
    return dataset, fit_rows


def _brute_force_scores(model, dataset, device, raw_user_id, candidate_ad_ids):
    """model.forward() run directly on a (user, ad) batch built from each side's own
    representative row, with TE_CROSS forced to 0 to match the scorer's documented
    simplification."""
    user_cache = UserFeatureCache(dataset)
    user_row = user_cache.row_for(raw_user_id)
    ad_ids_sorted, ad_rows_sorted = _last_row_per_ad(dataset)
    ad_row_lookup = dict(zip(ad_ids_sorted.tolist(), ad_rows_sorted.tolist(), strict=True))

    logits, cotrain_logits = [], []
    for ad_id in candidate_ad_ids:
        ad_row = ad_row_lookup[int(ad_id)]
        b = {key: torch.from_numpy(dataset.X[key][[user_row]].astype(np.int64)) for key, _ in USER_FEATS}
        for key, _ in AD_FEATS:
            b[key] = torch.from_numpy(dataset.X[key][[ad_row]].astype(np.int64))

        hc, hb, ht = history(
            np.array([user_row]), first=dataset.first, end=dataset.end, BC=dataset.BC, BB=dataset.BB,
            BT=dataset.BT, seq_len=dataset.seq_len,
        )
        b["hist_cate"] = torch.from_numpy(hc.astype(np.int64))
        b["hist_brand"] = torch.from_numpy(hb.astype(np.int64))
        b["hist_btag"] = torch.from_numpy(ht.astype(np.int64))

        te = np.zeros((1, 8), dtype=np.float32)
        te[0, TE_USER] = dataset.TE[user_row, TE_USER]
        te[0, TE_AD] = dataset.TE[ad_row, TE_AD]
        te[0, TE_CROSS] = 0.0  # documented serving-time simplification
        b["te"] = torch.from_numpy(te)

        b = {k: v.to(device) for k, v in b.items()}
        with torch.no_grad():
            out = model(b)
        logits.append(torch.sigmoid(out["logit"]).item())
        cotrain_logits.append(torch.sigmoid(out["cotrain_logit"]).item())
    return np.array(logits), np.array(cotrain_logits)


def test_cascade_matches_full_model_on_top_and_two_tower_elsewhere(tmp_path, patch_kaggle_csv, synthetic_params):
    dataset, fit_rows = _build_dataset(tmp_path, synthetic_params)
    device = get_device()
    model = ThreeTower(HP, dataset.sizes, dataset.seq_len).to(device).eval()

    ad_cache = build_ad_cache(model, dataset, device)
    user_cache = UserFeatureCache(dataset)
    scorer = TwoStageScorer(model, ad_cache, user_cache, device, CASCADE_TOP_FRAC)

    raw_user_id = int(dataset.user_ids[int(dataset.u[fit_rows[0]]) - 1])
    candidates = ad_cache["ad_ids"][: min(30, len(ad_cache["ad_ids"]))]
    out = scorer.score(raw_user_id, candidates)

    full_logits, cotrain_logits = _brute_force_scores(model, dataset, device, raw_user_id, candidates)

    idx = np.searchsorted(ad_cache["ad_ids"], candidates)
    with torch.no_grad():
        user_batch = {k: v.to(device) for k, v in user_cache.batch_for(raw_user_id).items()}
        u_vec = model.user(user_batch)
        ad_vecs = torch.from_numpy(ad_cache["ad_vecs"][idx]).to(device)
        stage1 = torch.sigmoid((u_vec * ad_vecs).sum(-1) + model.dot_bias).cpu().numpy()
    k = max(1, int(round(CASCADE_TOP_FRAC * len(candidates))))
    top_mask = np.zeros(len(candidates), dtype=bool)
    top_mask[np.argsort(-stage1)[:k]] = True

    np.testing.assert_allclose(out[top_mask], full_logits[top_mask], atol=1e-5)
    np.testing.assert_allclose(out[~top_mask], cotrain_logits[~top_mask], atol=1e-5)


def test_unknown_ad_id_raises_key_error(tmp_path, patch_kaggle_csv, synthetic_params):
    dataset, fit_rows = _build_dataset(tmp_path, synthetic_params)
    device = get_device()
    model = ThreeTower(HP, dataset.sizes, dataset.seq_len).to(device).eval()
    ad_cache = build_ad_cache(model, dataset, device)
    scorer = TwoStageScorer(model, ad_cache, UserFeatureCache(dataset), device, CASCADE_TOP_FRAC)
    raw_user_id = int(dataset.user_ids[int(dataset.u[fit_rows[0]]) - 1])

    try:
        scorer.score(raw_user_id, np.array([int(ad_cache["ad_ids"].max()) + 999_999]))
        raise AssertionError("expected KeyError for an ad id never seen in the ad cache")
    except KeyError:
        pass
