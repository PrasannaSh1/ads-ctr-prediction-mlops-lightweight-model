"""Time-ordered (CatBoost-style) click-rate encodings, ported verbatim from notebook section 4.4.

Smoothed past click rate of the user, ad, campaign, advertiser, brand, category and
user x category, plus log price.

A training row's click rate uses only clicks from *earlier* rows, so a row never sees its own
label. Validation rows use the statistics known at the end of the fit period, and the test day
uses all training days — so validation is encoded exactly like the test day.

Do NOT switch to out-of-fold encoding: leaving a row's fold out makes its encoded rate slightly
*lower* when the row was clicked (its own click is missing), so on the training rows click rates
become negatively related to clicks, while the test day — encoded from all training rows — has
the normal positive relation. Models learn the wrong direction and validation, encoded the same
way as training, does not notice. Development runs showed a validation-test AUC gap of about
0.025-0.027 partly for this reason. See PROMPT_taobao_ctr_mlops.md, constraint 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TE_COLS = ["te_user", "te_ad", "te_campaign", "te_customer", "te_brand", "te_cate", "te_user_cate", "log_price"]


def click_rate_features(
    imp: dict, X: dict, fit_rows: np.ndarray, val_rows: np.ndarray, test_rows: np.ndarray, n_cate: int,
    *, te_alpha: float,
) -> np.ndarray:
    y = imp["y"]
    prior, a = y[fit_rows].mean(), te_alpha
    train_rows = np.concatenate([fit_rows, val_rows])

    def rate_table(keys, rows):
        g = pd.DataFrame({"k": keys[rows], "y": y[rows]}).groupby("k")["y"].agg(["sum", "count"])
        return (g["sum"] + a * prior) / (g["count"] + a)

    def target_encode(keys):
        out = np.empty(len(keys), np.float32)
        # fit rows (time-sorted): clicks and views of the same key in EARLIER rows only
        k, yy = keys[fit_rows], y[fit_rows]
        g = pd.DataFrame({"k": k, "y": yy}).groupby("k")["y"]
        clicks_before, views_before = g.cumsum().values - yy, g.cumcount().values
        out[fit_rows] = (clicks_before + a * prior) / (views_before + a)
        # validation: what was known at the end of the fit period; test: what was known at the end of training
        out[val_rows] = pd.Series(keys[val_rows]).map(rate_table(keys, fit_rows)).fillna(prior).values
        out[test_rows] = pd.Series(keys[test_rows]).map(rate_table(keys, train_rows)).fillna(prior).values
        return out

    u = imp["u"]
    te = np.stack([
        target_encode(u), target_encode(X["adgroup_id"]), target_encode(X["campaign_id"]),
        target_encode(X["customer"]), target_encode(X["ad_brand"]), target_encode(X["ad_cate"]),
        target_encode(u * (n_cate + 1) + X["ad_cate"]),
        np.log1p(imp["price"]).astype(np.float32),
    ], 1)
    mu, sd = te[fit_rows].mean(0), te[fit_rows].std(0) + 1e-6
    return ((te - mu) / sd).astype(np.float32)  # standardised with fit-period statistics
