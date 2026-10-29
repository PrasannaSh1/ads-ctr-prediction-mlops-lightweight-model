"""history() gather must exactly match a brute-force rebuild straight from the raw CSV,
including the "no future leakage" requirement: only actions strictly before the impression are
ever included (notebook section 4, constraint 2 in PROMPT_taobao_ctr_mlops.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taobao_ctr.data.history import BTAG, build_history, build_vocab, history, stream_behaviour
from taobao_ctr.data.impressions import encode, load_impressions

SEQ_LEN = 10  # small on purpose: exercises both padding (few actions) and truncation (many actions)
N_BUCKETS = 4


def _brute_force_history(behavior_df, user, imp_time, seq_len, cate_vocab, brand_vocab, t_lo, t_hi):
    """Independent, obviously-correct (if slow) reimplementation: filter to this user's actions
    strictly before imp_time and inside the valid timestamp window, take the most recent
    seq_len, left-pad with zeros."""
    sub = behavior_df[
        (behavior_df["user"] == user)
        & (behavior_df["time_stamp"] < imp_time)
        & behavior_df["time_stamp"].between(t_lo, t_hi)
    ].sort_values("time_stamp", kind="stable").tail(seq_len)
    cates = encode(sub["cate"].values.astype(np.int64), cate_vocab)
    brands = encode(sub["brand"].fillna(-1).values.astype(np.int64), brand_vocab)
    btags = sub["btag"].astype(str).map(BTAG).fillna(0).astype(np.int8).values
    pad = seq_len - len(sub)
    cates = np.concatenate([np.zeros(pad, cates.dtype), cates])
    brands = np.concatenate([np.zeros(pad, brands.dtype), brands])
    btags = np.concatenate([np.zeros(pad, btags.dtype), btags])
    return cates, brands, btags


def test_history_matches_brute_force_rebuild(tmp_path, synthetic_data_dir, patch_kaggle_csv):
    imp = load_impressions(data_dir=tmp_path, kaggle_dataset="unused/unused")
    seen = stream_behaviour(imp, data_dir=tmp_path, kaggle_dataset="unused/unused", n_buckets=N_BUCKETS, chunk_rows=5_000)
    ad_cate = np.nan_to_num(imp["ad_cate"], nan=-1).astype(np.int64)
    ad_brand = np.nan_to_num(imp["ad_brand"], nan=-1).astype(np.int64)
    cate_vocab, brand_vocab = build_vocab(ad_cate, ad_brand, seen)
    first, end, BC, BB, BT = build_history(
        imp, cate_vocab, brand_vocab, data_dir=tmp_path, n_buckets=N_BUCKETS, seq_len=SEQ_LEN,
    )

    behavior_df = pd.read_csv(
        synthetic_data_dir / "behavior_log.csv" / "behavior_log.csv",
        dtype={"user": "int32", "time_stamp": "int64", "brand": "float64"},
    )
    t_lo, t_hi = imp["ts"].min() - 30 * 86_400, imp["ts"].max()

    rng = np.random.default_rng(0)
    sample_rows = rng.choice(len(imp["ts"]), size=min(200, len(imp["ts"])), replace=False)
    gBC, gBB, gBT = history(sample_rows, first=first, end=end, BC=BC, BB=BB, BT=BT, seq_len=SEQ_LEN)
    for i, row in enumerate(sample_rows):
        raw_user = imp["user_ids"][imp["u"][row] - 1]
        ec, eb, et = _brute_force_history(
            behavior_df, raw_user, imp["ts"][row], SEQ_LEN, cate_vocab, brand_vocab, t_lo, t_hi,
        )
        np.testing.assert_array_equal(gBC[i], ec, err_msg=f"category mismatch at row {row}")
        np.testing.assert_array_equal(gBB[i], eb, err_msg=f"brand mismatch at row {row}")
        np.testing.assert_array_equal(gBT[i], et, err_msg=f"action-type mismatch at row {row}")


