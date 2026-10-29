"""Generate the synthetic dataset and arrange it exactly as kaggle_csv() expects an
already-downloaded dataset to look: <data_dir>/<fname-without-extension>/<fname>, plus a
.complete marker. Local runs, CI and `dvc repro` call this once so the pipeline never touches
kagglehub or the real 24 GB dataset.

usage: python scripts/setup_synthetic_data.py [--data-dir DIR] [--impressions N] [--behaviour N] [--seed N]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_synthetic_data  # noqa: E402

FILES = ["raw_sample.csv", "ad_feature.csv", "user_profile.csv", "behavior_log.csv"]


def setup(data_dir: str, n_imp: int, n_beh: int, seed: int) -> None:
    staging = os.path.join(data_dir, "_synthetic_staging")
    make_synthetic_data.make(staging, n_imp=n_imp, n_beh=n_beh, seed=seed)
    for fname in FILES:
        folder = os.path.join(data_dir, fname.replace(".csv", ""))
        os.makedirs(folder, exist_ok=True)
        shutil.copy(os.path.join(staging, fname, fname), os.path.join(folder, fname))
        open(os.path.join(folder, ".complete"), "w").close()
    shutil.rmtree(staging)
    print(f"synthetic dataset ready at {data_dir} (kaggle_csv()-compatible layout)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("TAOBAO_DATA_DIR", "./data"))
    ap.add_argument("--impressions", type=int, default=4_000)
    ap.add_argument("--behaviour", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    setup(a.data_dir, n_imp=a.impressions, n_beh=a.behaviour, seed=a.seed)
