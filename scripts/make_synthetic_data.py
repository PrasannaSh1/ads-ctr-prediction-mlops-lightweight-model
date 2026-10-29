"""Make a small synthetic copy of the Taobao ad dataset (same files, columns and quirks as the real one).

Used for tests and CI: the real data is 24 GB and cannot run on a laptop or in GitHub Actions.
Clicks depend on (a) an ad-level click rate, (b) a user-level click rate and (c) whether the ad's
category is the user's favourite category, which also dominates the user's behaviour history. So the
models have real signal to learn, and a correct pipeline reaches a test AUC of roughly 0.63-0.65.

Layout written (identical to the Kaggle dataset `zerotrustai/taobao-ad-display-click`):
    <out>/raw_sample.csv/raw_sample.csv
    <out>/ad_feature.csv/ad_feature.csv
    <out>/user_profile.csv/user_profile.csv
    <out>/behavior_log.csv/behavior_log.csv

usage: python make_synthetic_data.py OUT_DIR [--impressions 200000] [--behaviour 1000000] [--seed 0]
"""
import argparse
import os

import numpy as np
import pandas as pd


def make(out_dir, n_users=3000, n_ads=800, n_imp=200_000, n_beh=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1494000000                                                       # 2017-05-05, as in the real data

    ads = pd.DataFrame({
        "adgroup_id": np.arange(1, n_ads + 1),
        "cate_id": rng.integers(1, 60, n_ads),
        "campaign_id": rng.integers(1, 300, n_ads),
        "customer": rng.integers(1, 200, n_ads),
        "brand": np.where(rng.random(n_ads) < 0.3, np.nan, rng.integers(1, 150, n_ads)),   # ~30% missing, like the real file
        "price": rng.lognormal(4, 1.5, n_ads),
    })
    users = pd.DataFrame({
        "userid": np.arange(1, n_users + 1),
        "cms_segid": rng.integers(0, 97, n_users),
        "cms_group_id": rng.integers(0, 13, n_users),
        "final_gender_code": rng.integers(1, 3, n_users),
        "age_level": rng.integers(0, 7, n_users),
        "pvalue_level": np.where(rng.random(n_users) < .5, np.nan, rng.integers(1, 4, n_users)),
        "shopping_level": rng.integers(1, 4, n_users),
        "occupation": rng.integers(0, 2, n_users),
        "new_user_class_level ": np.where(rng.random(n_users) < .3, np.nan, rng.integers(1, 5, n_users)),  # trailing space: real quirk
    }).iloc[:-200]                                                        # some users have no profile, like the real data

    u_imp = rng.integers(1, n_users + 1, n_imp)
    a_imp = rng.integers(1, n_ads + 1, n_imp)
    pref = rng.integers(1, 60, n_users + 1)                               # each user's favourite category
    ad_ctr = rng.lognormal(-3.4, 0.6, n_ads + 1)
    user_ctr = rng.lognormal(0, 0.5, n_users + 1)
    same_cate = ads.set_index("adgroup_id").loc[a_imp, "cate_id"].values == pref[u_imp]
    p_click = np.clip(ad_ctr[a_imp] * user_ctr[u_imp] * np.where(same_cate, 3.0, 1.0), 0, 0.9)
    clk = (rng.random(n_imp) < p_click).astype(int)
    raw = pd.DataFrame({
        "user": u_imp, "time_stamp": t0 + rng.integers(0, 8 * 86400, n_imp), "adgroup_id": a_imp,
        "pid": rng.choice(["430548_1007", "430539_1007"], n_imp), "nonclk": 1 - clk, "clk": clk,
    })

    ub = rng.integers(1, n_users + 400, n_beh)                            # some behaviour users never saw an ad
    beh = pd.DataFrame({
        "user": ub,
        "time_stamp": t0 - 14 * 86400 + rng.integers(0, 22 * 86400, n_beh),
        "btag": rng.choice(["pv", "pv", "pv", "cart", "fav", "buy"], n_beh),
        "cate": np.where(rng.random(n_beh) < 0.5, pref[np.clip(ub, 0, n_users)], rng.integers(1, 80, n_beh)),
        "brand": np.where(rng.random(n_beh) < 0.2, np.nan, rng.integers(1, 200, n_beh)),
    })
    beh.loc[:5, "time_stamp"] = 99                                        # corrupt timestamps, like the real file

    for name, df in [("raw_sample", raw), ("ad_feature", ads), ("user_profile", users), ("behavior_log", beh)]:
        folder = os.path.join(out_dir, f"{name}.csv")
        os.makedirs(folder, exist_ok=True)
        df.to_csv(os.path.join(folder, f"{name}.csv"), index=False)
    print(f"wrote synthetic dataset to {out_dir}: {n_imp:,} impressions, {n_beh:,} behaviour rows, CTR {clk.mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--impressions", type=int, default=200_000)
    ap.add_argument("--behaviour", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.out_dir, n_imp=a.impressions, n_beh=a.behaviour, seed=a.seed)
