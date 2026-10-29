"""Builds a tiny synthetic dataset once per test session (scripts/make_synthetic_data.py — the
same generator used for CI and local runs) and provides a drop-in replacement for kaggle_csv()
that points at it, so data/feature tests exercise the real pipeline code without touching the
network or the 24 GB real dataset.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_spec = importlib.util.spec_from_file_location(
    "make_synthetic_data", REPO_ROOT / "scripts" / "make_synthetic_data.py"
)
make_synthetic_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_synthetic_data)


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """A tiny synthetic copy of the dataset (same 4 files, columns and quirks as the real one),
    built once per test session so every test shares it instead of regenerating it."""
    out = tmp_path_factory.mktemp("synthetic_data")
    make_synthetic_data.make(str(out), n_users=300, n_ads=80, n_imp=4_000, n_beh=20_000, seed=0)
    return out


@pytest.fixture(scope="session")
def smoke_data_dir(tmp_path_factory) -> Path:
    """A larger synthetic copy, used only by the end-to-end AUC smoke test. The tiny
    `synthetic_data_dir` above is sparse enough (50 impressions/ad) that click-rate and
    embedding signal is dominated by noise and test-day AUC is unreliable; this uses
    scripts/make_synthetic_data.py's own defaults, which its docstring tunes to land a
    *correctly wired* pipeline around 0.63-0.65 test AUC."""
    out = tmp_path_factory.mktemp("smoke_data")
    make_synthetic_data.make(str(out), seed=0)  # n_users=3000, n_ads=800, n_imp=200_000, n_beh=1_000_000
    return out


def _fake_kaggle_csv_for(data_dir):
    def _fake(fname, **_ignored_kwargs):
        return str(data_dir / fname / fname)

    return _fake


@pytest.fixture
def fake_kaggle_csv(synthetic_data_dir):
    """Drop-in replacement for kaggle_csv() pointing straight at the synthetic files, matching
    the layout scripts/make_synthetic_data.py writes (<dir>/<fname>/<fname>) — kaggle_csv()'s
    own download/retry behaviour is tested separately in test_download.py."""
    return _fake_kaggle_csv_for(synthetic_data_dir)


@pytest.fixture
def patch_kaggle_csv(monkeypatch, fake_kaggle_csv):
    """Opt-in fixture: request it by name to make load_impressions()/stream_behaviour() read the
    synthetic dataset instead of trying to download anything."""
    monkeypatch.setattr("taobao_ctr.data.impressions.kaggle_csv", fake_kaggle_csv)
    monkeypatch.setattr("taobao_ctr.data.history.kaggle_csv", fake_kaggle_csv)


@pytest.fixture
def patch_kaggle_csv_smoke(monkeypatch, smoke_data_dir):
    """Same as patch_kaggle_csv, but points at the larger smoke_data_dir."""
    fake = _fake_kaggle_csv_for(smoke_data_dir)
    monkeypatch.setattr("taobao_ctr.data.impressions.kaggle_csv", fake)
    monkeypatch.setattr("taobao_ctr.data.history.kaggle_csv", fake)


@pytest.fixture
def synthetic_params():
    """params.yaml merged with params.synthetic.yaml -- the same settings CI/local runs use."""
    from taobao_ctr.config import load_params

    return load_params("synthetic")
