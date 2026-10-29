"""CLI entry point for every dvc.yaml stage: `python -m taobao_ctr.pipeline <stage> [--profile P] [--name NAME]`.

Each stage reads its inputs from `artifacts/cache`, computes, and writes its outputs back there
(plus `metrics/*.json` for anything dvc should track as a metric). DVC decides whether a stage
needs to rerun based on file hashes -- these functions always just (re)compute, they don't do
their own "already done" checks (Phase 2's stream_behaviour()/build_history() still do, for
resumability *within* a stage that dvc treats as a single unit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from taobao_ctr.config import AppConfig, load_config
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

def _cache(cfg: AppConfig) -> Path:
    return cfg.paths.artifact_dir / "cache"


# ---------------------------------------------------------------------------
# Stage 1: prepare_impressions
# ---------------------------------------------------------------------------

def stage_prepare_impressions(cfg: AppConfig) -> None:
    cache = _cache(cfg)
    cache.mkdir(parents=True, exist_ok=True)
    imp = load_impressions(
        data_dir=cfg.paths.data_dir, kaggle_dataset=cfg.params.data.kaggle_dataset, platform=cfg.platform,
    )
    arrays = {
        "ts": imp["ts"], "y": imp["y"], "u": imp["u"], "user_ids": imp["user_ids"], "price": imp["price"],
        "ad_cate_raw": imp["ad_cate"], "ad_brand_raw": imp["ad_brand"],
        **{f"x_{c}": imp["X"][c] for c in imp["X"]},
    }
    np.savez(cache / "impressions.npz", **arrays)
    (cache / "sizes_partial.json").write_text(json.dumps(imp["sizes"]))
    print(f"prepare_impressions: {len(imp['ts']):,} impressions, {len(imp['user_ids']):,} users")


def _load_impressions_cache(cache: Path) -> dict:
    a = np.load(cache / "impressions.npz")
    X = {c: a[f"x_{c}"] for c in ID_COLS if c not in ("ad_cate", "ad_brand")}
    return dict(
        ts=a["ts"], y=a["y"], u=a["u"], user_ids=a["user_ids"], price=a["price"],
        ad_cate=a["ad_cate_raw"], ad_brand=a["ad_brand_raw"], X=X,
        sizes=json.loads((cache / "sizes_partial.json").read_text()),
    )


# ---------------------------------------------------------------------------
# Stage 2: build_history
# ---------------------------------------------------------------------------

def stage_build_history(cfg: AppConfig) -> None:
    cache = _cache(cfg)
    imp = _load_impressions_cache(cache)
    ad_cate = np.nan_to_num(imp["ad_cate"], nan=-1).astype(np.int64)
    ad_brand = np.nan_to_num(imp["ad_brand"], nan=-1).astype(np.int64)
    seen = stream_behaviour(
        imp, data_dir=cfg.paths.data_dir, kaggle_dataset=cfg.params.data.kaggle_dataset, platform=cfg.platform,
        n_buckets=cfg.params.data.n_buckets, chunk_rows=cfg.params.data.chunk_rows,
    )
    cate_vocab, brand_vocab = build_vocab(ad_cate, ad_brand, seen)
    save_vocab(cache / "vocab.npz", cate_vocab, brand_vocab)

    first, end, BC, BB, BT = build_history(
        imp, cate_vocab, brand_vocab, data_dir=cfg.paths.data_dir,
        n_buckets=cfg.params.data.n_buckets, seq_len=cfg.params.data.seq_len,
    )
    save_checkpoint(cache / "history.npz", first, end, BC, BB, BT)

    sizes = dict(imp["sizes"])
    sizes["cate"], sizes["brand"] = len(cate_vocab), len(brand_vocab)
    sizes["price_bucket"], sizes["btag"] = 39, 4
    (cache / "sizes.json").write_text(json.dumps(sizes))
    np.savez(
        cache / "ad_features.npz",
        ad_cate=encode(ad_cate, cate_vocab).astype(np.int32), ad_brand=encode(ad_brand, brand_vocab).astype(np.int32),
    )
    print(f"build_history: {len(BT):,} actions in the compact store")


# ---------------------------------------------------------------------------
# Stage 3: click_rates
# ---------------------------------------------------------------------------

def stage_click_rates(cfg: AppConfig) -> None:
    cache = _cache(cfg)
    imp = _load_impressions_cache(cache)
    ad = np.load(cache / "ad_features.npz")
    sizes = json.loads((cache / "sizes.json").read_text())
    X = dict(imp["X"], ad_cate=ad["ad_cate"], ad_brand=ad["ad_brand"])

    train_rows, fit_rows, val_rows, test_rows = split(imp["ts"])
    np.savez(cache / "split.npz", train_rows=train_rows, fit_rows=fit_rows, val_rows=val_rows, test_rows=test_rows)

    TE = click_rate_features(imp, X, fit_rows, val_rows, test_rows, sizes["cate"], te_alpha=cfg.params.features.te_alpha)
    np.save(cache / "TE.npy", TE)
    print(f"click_rates: fit {len(fit_rows):,} | val {len(val_rows):,} | test {len(test_rows):,}")


def load_dataset(cfg: AppConfig) -> tuple[Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reassembles the Dataset + row splits from the cache files written by the 3 stages above."""
    cache = _cache(cfg)
    imp = _load_impressions_cache(cache)
    ad = np.load(cache / "ad_features.npz")
    sizes = json.loads((cache / "sizes.json").read_text())
    X = dict(imp["X"], ad_cate=ad["ad_cate"], ad_brand=ad["ad_brand"])
    first, end, BC, BB, BT = load_checkpoint(cache / "history.npz")
    TE = np.load(cache / "TE.npy")
    sp = np.load(cache / "split.npz")
    dataset = Dataset(
        X=X, y=imp["y"], ts=imp["ts"], u=imp["u"], user_ids=imp["user_ids"], TE=TE,
        first=first, end=end, BC=BC, BB=BB, BT=BT, sizes=sizes, seq_len=cfg.params.data.seq_len,
    )
    return dataset, sp["train_rows"], sp["fit_rows"], sp["val_rows"], sp["test_rows"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--profile", choices=["full", "synthetic"], default=None)
    parser.add_argument("--name", default=None, help="model name, for train_deep/tune stages")
    args = parser.parse_args()
    cfg = load_config(args.profile)
    cfg.paths.ensure()

    from taobao_ctr import pipeline_stages  # local import: keeps heavy deps (torch/lightgbm/mlflow) lazy

    pipeline_stages.run(args.stage, cfg, args.name)


if __name__ == "__main__":
    main()
