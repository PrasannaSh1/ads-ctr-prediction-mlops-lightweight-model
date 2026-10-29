"""LightGBM-only interaction-count features (notebook section 8): how often the user's recent
history touched the ad's category/brand, the length of that history, plus a handful of small
categoricals -- concatenated after the 8 click-rate features to make LightGBM's feature matrix.
"""
from __future__ import annotations

import numpy as np

from taobao_ctr.data.history import history
from taobao_ctr.data.impressions import CTX_COLS, USER_COLS
from taobao_ctr.features.click_rates import TE_COLS

LOWCARD = USER_COLS[1:] + CTX_COLS
LGB_FEATURES = TE_COLS + ["n_same_cate", "n_same_brand", "n_same_brand_strong", "hist_len"] + LOWCARD


def lgb_features(dataset, rows: np.ndarray) -> np.ndarray:
    hc, hb, ht = history(
        rows, first=dataset.first, end=dataset.end, BC=dataset.BC, BB=dataset.BB, BT=dataset.BT,
        seq_len=dataset.seq_len,
    )
    ac, ab = dataset.X["ad_cate"][rows, None], dataset.X["ad_brand"][rows, None]
    same_cate = (hc == ac) & (ac > 0)
    same_brand = (hb == ab) & (ab > 0)
    inter = np.stack([same_cate.sum(1), same_brand.sum(1), (same_brand & (ht > 1)).sum(1), (ht > 0).sum(1)], 1)
    cats = np.stack([dataset.X[c][rows] for c in LOWCARD], 1)
    return np.hstack([dataset.TE[rows], inter, cats]).astype(np.float64)  # LightGBM's Sequence input must be float64
