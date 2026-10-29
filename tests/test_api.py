"""FastAPI TestClient: /health, /model-info, and /score's response schema and ordering."""
from __future__ import annotations

from fastapi.testclient import TestClient

from taobao_ctr.data.dataset import build_dataset
from taobao_ctr.models.towers import ThreeTower
from taobao_ctr.serving.ad_cache import build_ad_cache
from taobao_ctr.serving.api import create_app
from taobao_ctr.serving.scorer import TwoStageScorer, UserFeatureCache
from taobao_ctr.training.trainer import get_device

HP = {"emb_dim": 4, "tower_dim": 8, "n_heads": 2, "n_query_tokens": 2, "dropout": 0.0}


def _make_client(tmp_path, synthetic_params):
    p = synthetic_params
    dataset, _train, fit_rows, _val, _test = build_dataset(
        data_dir=tmp_path, kaggle_dataset="unused/unused", n_buckets=p.data.n_buckets,
        chunk_rows=p.data.chunk_rows, seq_len=p.data.seq_len, te_alpha=p.features.te_alpha,
    )
    device = get_device()
    model = ThreeTower(HP, dataset.sizes, dataset.seq_len).to(device).eval()
    ad_cache = build_ad_cache(model, dataset, device)
    scorer = TwoStageScorer(model, ad_cache, UserFeatureCache(dataset), device, p.serve.cascade_top_frac)
    raw_user_id = int(dataset.user_ids[int(dataset.u[fit_rows[0]]) - 1])
    candidates = ad_cache["ad_ids"][: min(10, len(ad_cache["ad_ids"]))].tolist()
    return TestClient(create_app(scorer)), raw_user_id, candidates


def test_health(tmp_path, patch_kaggle_csv, synthetic_params):
    client, _user_id, _candidates = _make_client(tmp_path, synthetic_params)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info(tmp_path, patch_kaggle_csv, synthetic_params):
    client, _user_id, _candidates = _make_client(tmp_path, synthetic_params)
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "taobao-ctr-3tower"
    assert body["alias"] == "champion"
    assert body["n_ads_cached"] > 0


def test_score_schema_and_ordering(tmp_path, patch_kaggle_csv, synthetic_params):
    client, user_id, candidates = _make_client(tmp_path, synthetic_params)
    resp = client.post("/score", json={"user_id": user_id, "ad_ids": candidates})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == user_id
    returned_ids = {int(k) for k in body["scores"]}
    assert returned_ids == set(candidates)
    for score in body["scores"].values():
        assert 0.0 <= score <= 1.0


def test_score_unknown_ad_returns_404(tmp_path, patch_kaggle_csv, synthetic_params):
    client, user_id, candidates = _make_client(tmp_path, synthetic_params)
    bogus_ad_id = max(candidates) + 999_999
    resp = client.post("/score", json={"user_id": user_id, "ad_ids": [bogus_ad_id]})
    assert resp.status_code == 404
