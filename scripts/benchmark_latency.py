"""p50/p99 latency at 10K/50K/200K candidates for (a) two-tower-only scoring, (b) the two-stage
cascade scorer, (c) full 3-tower on every candidate (no cascade), (d) LightGBM with per-pair
feature computation. Same process, same hardware, same candidate set per cell; warm-up calls are
discarded and each cell is timed over multiple repeats. Writes reports/latency.md -- this
benchmark is the key interview evidence for why the 3-tower scales and LightGBM does not, so it
is kept honest rather than favorable.

usage: python scripts/benchmark_latency.py [--profile synthetic|full] [--repeats 20] [--warmup 3]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taobao_ctr.config import load_config  # noqa: E402
from taobao_ctr.features.interactions import lgb_features  # noqa: E402
from taobao_ctr.models.towers import ThreeTower  # noqa: E402
from taobao_ctr.pipeline import load_dataset  # noqa: E402
from taobao_ctr.serving.ad_cache import build_ad_cache  # noqa: E402
from taobao_ctr.serving.scorer import TwoStageScorer, UserFeatureCache  # noqa: E402
from taobao_ctr.training.lightgbm_train import train_lightgbm  # noqa: E402
from taobao_ctr.training.trainer import default_hp, get_device  # noqa: E402

CANDIDATE_COUNTS = [10_000, 50_000, 200_000]


def _percentiles(times_ms: list[float]) -> tuple[float, float]:
    arr = np.array(times_ms)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 99))


def _time_calls(fn, repeats: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return _percentiles(times)


def benchmark(cfg, repeats: int = 20, warmup: int = 3) -> list[dict]:
    dataset, _train, fit_rows, val_rows, _test = load_dataset(cfg)
    device = get_device()
    model = ThreeTower(default_hp(cfg.params), dataset.sizes, dataset.seq_len).to(device).eval()

    ad_cache = build_ad_cache(model, dataset, device)
    all_ad_ids = ad_cache["ad_ids"]
    user_cache = UserFeatureCache(dataset)
    raw_user_id = int(dataset.user_ids[int(dataset.u[fit_rows[0]]) - 1])
    scorer = TwoStageScorer(model, ad_cache, user_cache, device, cfg.params.serve.cascade_top_frac)

    lgb_params = {"objective": "binary", "metric": "auc", "verbose": -1, "seed": cfg.params.seed, **{
        k: v for k, v in cfg.params.train.lightgbm.model_dump().items()
        if k not in ("num_boost_round", "early_stopping_rounds", "log_evaluation_every")
    }}
    gbm = train_lightgbm(dataset, fit_rows, val_rows, lgb_params, num_boost_round=20, early_stopping_rounds=20, log_evaluation_every=0)

    rng = np.random.default_rng(0)
    results = []
    for n in CANDIDATE_COUNTS:
        # Sample with replacement: the point is to benchmark at this many *candidates per
        # request*, independent of how many distinct ads the (possibly tiny synthetic) corpus
        # has -- duplicates cost exactly as much to score as distinct ads would.
        n_eff = n
        candidates = rng.choice(all_ad_ids, size=n_eff, replace=True)
        idx = np.searchsorted(all_ad_ids, candidates)
        user_batch = {k: v.to(device) for k, v in user_cache.batch_for(raw_user_id).items()}
        u_vec = model.user(user_batch)

        def two_tower_only(idx=idx, u_vec=u_vec):
            with torch.no_grad():
                ad_vecs = torch.from_numpy(ad_cache["ad_vecs"][idx]).to(device)
                torch.sigmoid((u_vec * ad_vecs).sum(-1) + model.dot_bias)

        def two_stage(candidates=candidates):
            scorer.score(raw_user_id, candidates)

        def full_three_tower(candidates=candidates):
            saved = scorer.cascade_top_frac
            scorer.cascade_top_frac = 1.0
            try:
                scorer.score(raw_user_id, candidates)
            finally:
                scorer.cascade_top_frac = saved

        # LightGBM needs a real (user, ad) impression row per candidate pair for its per-pair
        # interaction features -- cycle through fit_rows to synthesize n_eff such pairs.
        rows = fit_rows[np.arange(n_eff) % len(fit_rows)]

        def lgb_scoring(rows=rows):
            feats = lgb_features(dataset, rows)
            gbm.predict(feats, num_iteration=gbm.best_iteration)

        for method, fn in [
            ("two-tower only", two_tower_only),
            ("two-stage scorer", two_stage),
            ("full 3-tower (no cascade)", full_three_tower),
            ("LightGBM (per-pair features)", lgb_scoring),
        ]:
            p50, p99 = _time_calls(fn, repeats, warmup)
            results.append({"candidates": n_eff, "method": method, "p50_ms": round(p50, 2), "p99_ms": round(p99, 2)})
            print(f"{n_eff:>7,} candidates | {method:<30} | p50 {p50:8.2f} ms | p99 {p99:8.2f} ms")
    return results


def render_report(results: list[dict], repeats: int, warmup: int) -> str:
    df = pd.DataFrame(results)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    table = "\n".join([header, sep, *rows])
    return (
        "# Latency benchmark\n\n"
        f"Same process and hardware for every row; {warmup} warm-up call(s) discarded, "
        f"{repeats} timed repeats per cell.\n\n" + table + "\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=["synthetic", "full"], default="synthetic")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    results = benchmark(cfg, repeats=args.repeats, warmup=args.warmup)
    out = Path("reports") / "latency.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(results, args.repeats, args.warmup))
    print(f"wrote {out}")
