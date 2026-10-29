"""Precompute ad-tower vectors and cross-tower query projections for every ad once, offline
(notebook section 6, AdTower docstring: "can be computed offline and cached per ad"). The
scorer then only runs the user tower per request and does a cheap dot product / attention over
these cached vectors instead of recomputing the ad side for every request.

Simplification: an ad's features/click-rates are taken from its most recent impression row in
the dataset, not re-aggregated online -- a real feature store would refresh these continuously.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from taobao_ctr.data.dataset import Dataset
from taobao_ctr.models.layers import AD_FEATS
from taobao_ctr.models.towers import ThreeTower


def _last_row_per_ad(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Encoded ad ids (1..K, ascending) and the row index of each ad's most recent impression."""
    ad_ids = dataset.X["adgroup_id"]
    order = np.argsort(dataset.ts, kind="stable")
    last_row: dict[int, int] = {}
    for idx in order:  # later rows overwrite earlier ones -> ends up with the most recent per ad
        last_row[int(ad_ids[idx])] = int(idx)
    unique_ids = np.array(sorted(last_row))
    rows = np.array([last_row[a] for a in unique_ids])
    return unique_ids, rows


def build_ad_cache(model: ThreeTower, dataset: Dataset, device: torch.device, batch_size: int = 4096) -> dict:
    """Returns {"ad_ids": (K,), "ad_vecs": (K, tower_dim), "cross_vecs": (K, tower_dim)}."""
    ad_ids, rows = _last_row_per_ad(dataset)
    model.to(device).eval()
    ad_vecs, cross_vecs = [], []
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            b = {key: torch.from_numpy(dataset.X[key][chunk].astype(np.int64)).to(device) for key, _ in AD_FEATS}
            b["te"] = torch.from_numpy(dataset.TE[chunk]).to(device)
            vec, cross = model.ad(b)
            ad_vecs.append(vec.cpu().numpy())
            cross_vecs.append(cross.cpu().numpy())
    return {"ad_ids": ad_ids, "ad_vecs": np.concatenate(ad_vecs), "cross_vecs": np.concatenate(cross_vecs)}


def save_ad_cache(path: str | Path, cache: dict) -> None:
    np.savez(path, ad_ids=cache["ad_ids"], ad_vecs=cache["ad_vecs"], cross_vecs=cache["cross_vecs"])


def load_ad_cache(path: str | Path) -> dict:
    a = np.load(path)
    return {"ad_ids": a["ad_ids"], "ad_vecs": a["ad_vecs"], "cross_vecs": a["cross_vecs"]}
