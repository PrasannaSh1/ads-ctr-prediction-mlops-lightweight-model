"""Click-rate features (also called target encoding), computed in time order.

For every impression we compute the smoothed past click rate of its user, ad, campaign,
advertiser, brand, category and user x category, plus the log of the price. That is 8 features.

Rules that stop label leakage:
- A fit row only uses clicks from EARLIER rows, so a row never sees its own label.
- Validation rows use the rates known at the end of the fit period.
- Test rows use the rates from all training days (fit + validation).
So validation is encoded the same way as the test day.

Do NOT switch to out-of-fold encoding. It makes a clicked row's own rate slightly lower (its own
click is missing), so on the training rows the rate becomes negatively related to clicks while
the test day keeps the normal positive relation. Models learn the wrong direction and validation
does not notice. In development this caused a validation-test AUC gap of about 0.025-0.027.
"""
import numpy as np
import pandas as pd

TE_COLS = ["te_user", "te_ad", "te_campaign", "te_customer", "te_brand", "te_cate", "te_user_cate", "log_price"]


def rate_table(keys, y, rows, prior, alpha):
    """Smoothed click rate per key, computed from the given rows only.

    rate = (clicks + alpha * prior) / (views + alpha)
    Returns a pandas Series: key -> rate.
    """
    df = pd.DataFrame({"k": keys[rows], "y": y[rows]})
    grouped = df.groupby("k")["y"].agg(["sum", "count"])
    return (grouped["sum"] + alpha * prior) / (grouped["count"] + alpha)


def target_encode(keys, y, fit_rows, val_rows, test_rows, prior, alpha):
    """One click-rate feature for every impression, following the leakage rules above."""
    out = np.empty(len(keys), np.float32)

    # Fit rows are in time order. For each row: clicks and views of the same key in EARLIER rows only.
    fit_keys = keys[fit_rows]
    fit_y = y[fit_rows]
    grouped = pd.DataFrame({"k": fit_keys, "y": fit_y}).groupby("k")["y"]
    clicks_before = grouped.cumsum().values - fit_y  # running total minus this row's own label
    views_before = grouped.cumcount().values  # how many earlier rows had this key
    out[fit_rows] = (clicks_before + alpha * prior) / (views_before + alpha)

    # Validation rows: the rates known at the end of the fit period. Unknown key -> prior.
    fit_table = rate_table(keys, y, fit_rows, prior, alpha)
    out[val_rows] = pd.Series(keys[val_rows]).map(fit_table).fillna(prior).values

    # Test rows: the rates from all training days (fit + validation). Unknown key -> prior.
    train_rows = np.concatenate([fit_rows, val_rows])
    train_table = rate_table(keys, y, train_rows, prior, alpha)
    out[test_rows] = pd.Series(keys[test_rows]).map(train_table).fillna(prior).values
    return out


def click_rate_features(imp, X, fit_rows, val_rows, test_rows, n_cate, te_alpha):
    """The 8 click-rate features for every impression, standardised with fit-period statistics."""
    y = imp["y"]
    prior = y[fit_rows].mean()  # the overall click rate of the fit period
    alpha = te_alpha  # smoothing: how many "virtual" views at the prior rate every key starts with
    u = imp["u"]

    # user x category: one key per (user, category) pair
    user_cate_key = u * (n_cate + 1) + X["ad_cate"]

    columns = [
        target_encode(u, y, fit_rows, val_rows, test_rows, prior, alpha),
        target_encode(X["adgroup_id"], y, fit_rows, val_rows, test_rows, prior, alpha),
        target_encode(X["campaign_id"], y, fit_rows, val_rows, test_rows, prior, alpha),
        target_encode(X["customer"], y, fit_rows, val_rows, test_rows, prior, alpha),
        target_encode(X["ad_brand"], y, fit_rows, val_rows, test_rows, prior, alpha),
        target_encode(X["ad_cate"], y, fit_rows, val_rows, test_rows, prior, alpha),
        target_encode(user_cate_key, y, fit_rows, val_rows, test_rows, prior, alpha),
        np.log1p(imp["price"]).astype(np.float32),
    ]
    te = np.stack(columns, 1)

    # Standardise every column: subtract the mean and divide by the standard deviation,
    # both measured on the fit rows only
    mean = te[fit_rows].mean(0)
    std = te[fit_rows].std(0) + 1e-6
    te = (te - mean) / std
    return te.astype(np.float32)
