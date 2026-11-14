"""The command line entry point. Every stage in dvc.yaml runs:

    python -m taobao_ctr.pipeline <stage> [--name "<model name>"]

This file holds the 3 data stages (prepare_impressions, build_history, click_rates) and
load_dataset(), which reads their output back. The other 7 stages are in pipeline_stages.py.
Every stage reads from and writes to artifacts/cache. DVC decides which stages must rerun.
"""
import argparse
import json
import os

import numpy as np

from taobao_ctr.config import load_config, make_folders
from taobao_ctr.data.dataset import Dataset
from taobao_ctr.data.history import (
    build_history,
    build_vocab,
    load_checkpoint,
    save_checkpoint,
    save_vocab,
    stream_behaviour,
)
from taobao_ctr.data.impressions import ID_COLS, encode, load_impressions, split
from taobao_ctr.features.click_rates import click_rate_features

STAGES = [
    "prepare_impressions",
    "build_history",
    "click_rates",
    "train_lightgbm",
    "train_deep",
    "tune",
    "evaluate",
    "combine",
    "register",
    "monitor",
]


def cache_folder(config):
    return os.path.join(config["paths"]["artifact_dir"], "cache")


# ---------------------------------------------------------------------------
# Stage 1: prepare_impressions
# ---------------------------------------------------------------------------

def stage_prepare_impressions(config):
    """Load the 3 CSVs, encode the ids, and save everything as numpy arrays."""
    cache = cache_folder(config)
    os.makedirs(cache, exist_ok=True)
    imp = load_impressions(config["paths"]["data_dir"], config["data"]["kaggle_dataset"])

    arrays = {
        "ts": imp["ts"],
        "y": imp["y"],
        "u": imp["u"],
        "user_ids": imp["user_ids"],
        "price": imp["price"],
        "ad_cate_raw": imp["ad_cate"],  # not encoded yet: stage 2 needs the raw values
        "ad_brand_raw": imp["ad_brand"],
    }
    for col in imp["X"]:
        arrays["x_" + col] = imp["X"][col]  # every encoded id column, with an x_ prefix
    np.savez(os.path.join(cache, "impressions.npz"), **arrays)

    with open(os.path.join(cache, "sizes_partial.json"), "w") as f:
        json.dump(imp["sizes"], f)
    print(f"prepare_impressions: {len(imp['ts']):,} impressions, {len(imp['user_ids']):,} users")


def load_impressions_cache(cache):
    """Read impressions.npz back into the same dict shape that load_impressions() returns."""
    a = np.load(os.path.join(cache, "impressions.npz"))
    X = {}
    for col in ID_COLS:
        if col == "ad_cate" or col == "ad_brand":
            continue  # these two are encoded in stage 2
        X[col] = a["x_" + col]
    with open(os.path.join(cache, "sizes_partial.json")) as f:
        sizes = json.load(f)
    imp = {
        "ts": a["ts"],
        "y": a["y"],
        "u": a["u"],
        "user_ids": a["user_ids"],
        "price": a["price"],
        "ad_cate": a["ad_cate_raw"],
        "ad_brand": a["ad_brand_raw"],
        "X": X,
        "sizes": sizes,
    }
    return imp


# ---------------------------------------------------------------------------
# Stage 2: build_history
# ---------------------------------------------------------------------------

def stage_build_history(config):
    """Stream the behaviour log into buckets, build the compact history store and the vocab."""
    cache = cache_folder(config)
    data_dir = config["paths"]["data_dir"]
    n_buckets = config["data"]["n_buckets"]
    imp = load_impressions_cache(cache)

    # Missing category / brand (NaN) becomes -1, which encode() turns into 0 (unknown)
    ad_cate = np.nan_to_num(imp["ad_cate"], nan=-1).astype(np.int64)
    ad_brand = np.nan_to_num(imp["ad_brand"], nan=-1).astype(np.int64)

    seen = stream_behaviour(imp, data_dir, config["data"]["kaggle_dataset"], n_buckets, config["data"]["chunk_rows"])
    cate_vocab, brand_vocab = build_vocab(ad_cate, ad_brand, seen)
    save_vocab(os.path.join(cache, "vocab.npz"), cate_vocab, brand_vocab)

    first, end, BC, BB, BT = build_history(imp, cate_vocab, brand_vocab, data_dir, n_buckets, config["data"]["seq_len"])
    save_checkpoint(os.path.join(cache, "history.npz"), first, end, BC, BB, BT)

    # The complete vocabulary sizes: stage 1's sizes plus category, brand, price bucket and action type
    sizes = dict(imp["sizes"])
    sizes["cate"] = len(cate_vocab)
    sizes["brand"] = len(brand_vocab)
    sizes["price_bucket"] = 39
    sizes["btag"] = 4
    with open(os.path.join(cache, "sizes.json"), "w") as f:
        json.dump(sizes, f)

    np.savez(
        os.path.join(cache, "ad_features.npz"),
        ad_cate=encode(ad_cate, cate_vocab).astype(np.int32),
        ad_brand=encode(ad_brand, brand_vocab).astype(np.int32),
    )
    print(f"build_history: {len(BT):,} actions in the compact store")


# ---------------------------------------------------------------------------
# Stage 3: click_rates
# ---------------------------------------------------------------------------

def stage_click_rates(config):
    """Split the rows by time and compute the 8 click-rate features."""
    cache = cache_folder(config)
    imp = load_impressions_cache(cache)
    ad = np.load(os.path.join(cache, "ad_features.npz"))
    with open(os.path.join(cache, "sizes.json")) as f:
        sizes = json.load(f)
    X = dict(imp["X"])
    X["ad_cate"] = ad["ad_cate"]
    X["ad_brand"] = ad["ad_brand"]

    train_rows, fit_rows, val_rows, test_rows = split(imp["ts"])
    np.savez(os.path.join(cache, "split.npz"), train_rows=train_rows, fit_rows=fit_rows, val_rows=val_rows,
             test_rows=test_rows)

    TE = click_rate_features(imp, X, fit_rows, val_rows, test_rows, sizes["cate"], config["features"]["te_alpha"])
    np.save(os.path.join(cache, "TE.npy"), TE)
    print(f"click_rates: fit {len(fit_rows):,} | val {len(val_rows):,} | test {len(test_rows):,}")


def load_dataset(config):
    """Put the output of the 3 data stages together into one Dataset, plus the 4 row splits."""
    cache = cache_folder(config)
    imp = load_impressions_cache(cache)
    ad = np.load(os.path.join(cache, "ad_features.npz"))
    with open(os.path.join(cache, "sizes.json")) as f:
        sizes = json.load(f)
    X = dict(imp["X"])
    X["ad_cate"] = ad["ad_cate"]
    X["ad_brand"] = ad["ad_brand"]
    first, end, BC, BB, BT = load_checkpoint(os.path.join(cache, "history.npz"))
    TE = np.load(os.path.join(cache, "TE.npy"))
    sp = np.load(os.path.join(cache, "split.npz"))
    dataset = Dataset(X, imp["y"], imp["ts"], imp["u"], imp["user_ids"], TE, first, end, BC, BB, BT, sizes,
                      config["data"]["seq_len"])
    return dataset, sp["train_rows"], sp["fit_rows"], sp["val_rows"], sp["test_rows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--name", default=None, help="model name, for the train_deep and tune stages")
    args = parser.parse_args()

    config = load_config()
    make_folders(config)

    # Imported here and not at the top: pipeline_stages imports this file (for load_dataset),
    # so importing it at the top would be a circular import. It also keeps --help fast.
    from taobao_ctr import pipeline_stages

    pipeline_stages.run(args.stage, config, args.name)


if __name__ == "__main__":
    main()
