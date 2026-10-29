"""Resumability: a build that stops after streaming the behaviour log, or after building the
history store, must pick up exactly where it left off on the next run, producing byte-identical
arrays to a run that never stopped (notebook section 4.5, constraint 7 in
PROMPT_taobao_ctr_mlops.md)."""
from __future__ import annotations

import numpy as np

from taobao_ctr.data.history import (
    build_history,
    build_vocab,
    history,
    load_checkpoint,
    load_vocab,
    save_checkpoint,
    save_vocab,
    stream_behaviour,
)
from taobao_ctr.data.impressions import load_impressions

N_BUCKETS = 4
SEQ_LEN = 10


def test_stream_behaviour_skips_rework_once_done_marker_exists(tmp_path, patch_kaggle_csv):
    imp = load_impressions(data_dir=tmp_path, kaggle_dataset="unused/unused")
    kwargs = dict(data_dir=tmp_path, kaggle_dataset="unused/unused", n_buckets=N_BUCKETS, chunk_rows=5_000)
    seen1 = stream_behaviour(imp, **kwargs)
    cate1, brand1 = np.array(seen1["cate"]), np.array(seen1["brand"])

    # A second call with a deliberately different `imp` would produce a different result if it
    # re-streamed; the done.npz marker must make it short-circuit and return the first result.
    different_imp = {**imp, "user_ids": np.array([999_999])}
    seen2 = stream_behaviour(different_imp, **kwargs)
    np.testing.assert_array_equal(seen2["cate"], cate1)
    np.testing.assert_array_equal(seen2["brand"], brand1)


def test_history_checkpoint_round_trip_is_identical(tmp_path, patch_kaggle_csv):
    imp = load_impressions(data_dir=tmp_path, kaggle_dataset="unused/unused")
    seen = stream_behaviour(imp, data_dir=tmp_path, kaggle_dataset="unused/unused", n_buckets=N_BUCKETS, chunk_rows=5_000)
    ad_cate = np.nan_to_num(imp["ad_cate"], nan=-1).astype(np.int64)
    ad_brand = np.nan_to_num(imp["ad_brand"], nan=-1).astype(np.int64)
    cate_vocab, brand_vocab = build_vocab(ad_cate, ad_brand, seen)
    first, end, BC, BB, BT = build_history(
        imp, cate_vocab, brand_vocab, data_dir=tmp_path, n_buckets=N_BUCKETS, seq_len=SEQ_LEN,
    )

    hist_path, vocab_path = tmp_path / "history.npz", tmp_path / "vocab.npz"
    save_checkpoint(hist_path, first, end, BC, BB, BT)
    save_vocab(vocab_path, cate_vocab, brand_vocab)

    r_first, r_end, r_BC, r_BB, r_BT = load_checkpoint(hist_path)
    r_cate_vocab, r_brand_vocab = load_vocab(vocab_path)

    np.testing.assert_array_equal(r_first, first)
    np.testing.assert_array_equal(r_end, end)
    np.testing.assert_array_equal(r_BC, BC)
    np.testing.assert_array_equal(r_BB, BB)
    np.testing.assert_array_equal(r_BT, BT)
    np.testing.assert_array_equal(r_cate_vocab, cate_vocab)
    np.testing.assert_array_equal(r_brand_vocab, brand_vocab)

    # the resumed (loaded) arrays must gather identically to the freshly-built ones
    rows = np.arange(min(50, len(first)))
    gc1, gb1, gt1 = history(rows, first=first, end=end, BC=BC, BB=BB, BT=BT, seq_len=SEQ_LEN)
    gc2, gb2, gt2 = history(rows, first=r_first, end=r_end, BC=r_BC, BB=r_BB, BT=r_BT, seq_len=SEQ_LEN)
    np.testing.assert_array_equal(gc1, gc2)
    np.testing.assert_array_equal(gb1, gb2)
    np.testing.assert_array_equal(gt1, gt2)
