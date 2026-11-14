"""Extra features for LightGBM only. Trees cannot read a list of 50 actions, so we count:
how often the user's recent history touched the ad's category / brand, and how long the
history is. These are added after the 8 click-rate features and 9 small id columns."""
import numpy as np

from taobao_ctr.data.history import get_history
from taobao_ctr.data.impressions import CTX_COLS, USER_COLS
from taobao_ctr.features.click_rates import TE_COLS

# Small id columns that LightGBM gets as categorical features (cms_segid is left out)
LOWCARD = USER_COLS[1:] + CTX_COLS
# The names of all 21 LightGBM features, in order
LGB_FEATURES = TE_COLS + ["n_same_cate", "n_same_brand", "n_same_brand_strong", "hist_len"] + LOWCARD


def lgb_features(dataset, rows):
    """The LightGBM feature matrix for the given rows: shape (len(rows), 21)."""
    hist_cate, hist_brand, hist_btag = get_history(dataset, rows)
    ad_cate = dataset.X["ad_cate"][rows, None]
    ad_brand = dataset.X["ad_brand"][rows, None]

    # For each past action: is it the same category / brand as the ad?
    same_cate = (hist_cate == ad_cate) & (ad_cate > 0)
    same_brand = (hist_brand == ad_brand) & (ad_brand > 0)

    n_same_cate = same_cate.sum(1)
    n_same_brand = same_brand.sum(1)
    n_same_brand_strong = (same_brand & (hist_btag > 1)).sum(1)  # same brand with cart / fav / buy
    hist_len = (hist_btag > 0).sum(1)  # how many real (non padding) actions
    counts = np.stack([n_same_cate, n_same_brand, n_same_brand_strong, hist_len], 1)

    cat_columns = []
    for col in LOWCARD:
        cat_columns.append(dataset.X[col][rows])
    cats = np.stack(cat_columns, 1)

    features = np.hstack([dataset.TE[rows], counts, cats])
    return features.astype(np.float64)  # LightGBM's Sequence input must be float64
