"""Load params.yaml (+ params.synthetic.yaml override), detect the run platform, and resolve paths.

Every other module asks this file two questions: "what are the settings?" and "where do things
live?". It mirrors the notebook's `CFG` dict and `ON_KAGGLE` / `ON_COLAB` flags, but as typed,
validated objects instead of a loose dict, with paths coming from the environment instead of
being hard-coded.
"""
from __future__ import annotations

import os
import sys
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class Platform(StrEnum):
    KAGGLE = "kaggle"
    COLAB = "colab"
    LOCAL = "local"


def detect_platform() -> Platform:
    if os.path.exists("/kaggle/input"):
        return Platform.KAGGLE
    if "google.colab" in sys.modules:
        return Platform.COLAB
    return Platform.LOCAL


class DataConfig(BaseModel):
    kaggle_dataset: str
    seq_len: int
    n_buckets: int
    chunk_rows: int


class FeaturesConfig(BaseModel):
    te_alpha: float


class ModelConfig(BaseModel):
    emb_dim: int
    tower_dim: int
    n_heads: int
    n_query_tokens: int


class LightGBMConfig(BaseModel):
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    num_boost_round: int
    early_stopping_rounds: int
    log_evaluation_every: int


class TrainConfig(BaseModel):
    lr: float
    batch_size: int
    dropout: float
    optimizer: str
    weight_decay: float
    loss: str
    gamma: float
    pos_weight: float
    cotrain_weight: float
    max_epochs: int
    eval_every_rows: int
    patience: int
    lightgbm: LightGBMConfig


class TuneConfig(BaseModel):
    tune_models: list[str]
    tune_rows: int
    tune_val_rows: int
    tune_trials: int
    tune_timeout_min: float


class ServeConfig(BaseModel):
    cascade_top_frac: float = Field(gt=0, le=1)


class Params(BaseModel):
    """Validated contents of params.yaml (merged with params.synthetic.yaml when profile="synthetic")."""

    seed: int
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    train: TrainConfig
    tune: TuneConfig
    serve: ServeConfig


class Paths(BaseModel):
    """Where raw data, caches/predictions/models and MLflow live, resolved for the current platform."""

    data_dir: Path
    artifact_dir: Path
    mlflow_tracking_uri: str

    def ensure(self) -> Paths:
        """Create the directory tree (data_dir, artifact_dir and its cache/preds/models subfolders)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("", "cache", "preds", "models"):
            (self.artifact_dir / sub).mkdir(parents=True, exist_ok=True)
        return self


def _default_data_dir(platform: Platform) -> str:
    # Fast local disk for raw CSVs and temporary buckets — does not need to survive a disconnect.
    return {
        Platform.KAGGLE: "/tmp/data",
        Platform.COLAB: "/content/data",
        Platform.LOCAL: "./data",
    }[platform]


def _default_artifact_dir(platform: Platform) -> str:
    # Caches, predictions, results and MLflow artifacts — ideally survives a disconnect/restart.
    return {
        Platform.KAGGLE: "/kaggle/working/taobao_final",
        Platform.COLAB: "/content/drive/MyDrive/taobao_final",
        Platform.LOCAL: "./artifacts",
    }[platform]


def resolve_paths(platform: Platform | None = None) -> Paths:
    """TAOBAO_DATA_DIR / TAOBAO_ARTIFACT_DIR / MLFLOW_TRACKING_URI win when set; otherwise a
    platform-appropriate default is used, matching the notebook's local_dir/drive_dir split."""
    platform = platform or detect_platform()
    data_dir = os.environ.get("TAOBAO_DATA_DIR", _default_data_dir(platform))
    artifact_dir = os.environ.get("TAOBAO_ARTIFACT_DIR", _default_artifact_dir(platform))
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    return Paths(data_dir=Path(data_dir), artifact_dir=Path(artifact_dir), mlflow_tracking_uri=mlflow_uri)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_params(profile: str | None = None) -> Params:
    """Load params.yaml, merging params.synthetic.yaml on top when profile="synthetic".

    profile defaults to the TAOBAO_PROFILE env var, then "full". Tests and CI set
    TAOBAO_PROFILE=synthetic (or pass profile="synthetic" directly) to run on tiny settings
    without touching any of the pipeline logic.
    """
    profile = profile or os.environ.get("TAOBAO_PROFILE", "full")
    with open(REPO_ROOT / "params.yaml") as f:
        raw = yaml.safe_load(f)
    if profile == "synthetic":
        with open(REPO_ROOT / "params.synthetic.yaml") as f:
            override = yaml.safe_load(f)
        raw = _deep_merge(raw, override)
    elif profile != "full":
        raise ValueError(f"unknown profile {profile!r}, expected 'full' or 'synthetic'")
    return Params.model_validate(raw)


class AppConfig(BaseModel):
    """Everything a pipeline stage needs: validated settings + resolved paths + the platform they came from."""

    params: Params
    paths: Paths
    platform: Platform


def load_config(profile: str | None = None) -> AppConfig:
    platform = detect_platform()
    return AppConfig(params=load_params(profile), paths=resolve_paths(platform), platform=platform)
