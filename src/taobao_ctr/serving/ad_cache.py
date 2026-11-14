"""The ad cache: run the ad tower once for every ad and keep its two output vectors.

At request time the scorer only runs the user tower and uses these cached vectors, instead of
running the ad side again for every request.

Simplification: an ad's features and click rates are taken from its most recent impression in
the dataset. A real system would refresh them continuously from a feature store.
"""
import numpy as np
import torch

from taobao_ctr.models.layers import AD_FEATS


def last_row_per_ad(dataset):
    """For every ad: its encoded id and the row of its most recent impression."""
    ad_ids = dataset.X["adgroup_id"]
    order = np.argsort(dataset.ts, kind="stable")
    last_row = {}
    for idx in order:  # later rows overwrite earlier ones, so we end with the most recent row per ad
        last_row[int(ad_ids[idx])] = int(idx)
    unique_ids = np.array(sorted(last_row))
    rows = []
    for ad_id in unique_ids:
        rows.append(last_row[ad_id])
    return unique_ids, np.array(rows)


def build_ad_cache(model, dataset, device, batch_size=4096):
    """Returns {"ad_ids": (K,), "ad_vecs": (K, tower_dim), "cross_vecs": (K, tower_dim)}."""
    ad_ids, rows = last_row_per_ad(dataset)
    model.to(device).eval()
    ad_vecs = []
    cross_vecs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = {}
            for batch_key, _table_name in AD_FEATS:
                batch[batch_key] = torch.from_numpy(dataset.X[batch_key][chunk].astype(np.int64)).to(device)
            batch["te"] = torch.from_numpy(dataset.TE[chunk]).to(device)
            ad_vector, cross_query = model.ad(batch)
            ad_vecs.append(ad_vector.cpu().numpy())
            cross_vecs.append(cross_query.cpu().numpy())
    return {"ad_ids": ad_ids, "ad_vecs": np.concatenate(ad_vecs), "cross_vecs": np.concatenate(cross_vecs)}


def save_ad_cache(path, cache):
    np.savez(path, ad_ids=cache["ad_ids"], ad_vecs=cache["ad_vecs"], cross_vecs=cache["cross_vecs"])


def load_ad_cache(path):
    a = np.load(path)
    return {"ad_ids": a["ad_ids"], "ad_vecs": a["ad_vecs"], "cross_vecs": a["cross_vecs"]}
