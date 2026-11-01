"""Loads params.yaml into typed, validated settings."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    data_dir: Path
    artifact_dir: Path
    mlflow_tracking_uri: str

    def ensure(self) -> PathsConfig:
        """Create data_dir and artifact_dir (with its cache/preds/models subfolders)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("", "cache", "preds", "models"):
            (self.artifact_dir / sub).mkdir(parents=True, exist_ok=True)
        return self


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


class Config(BaseModel):
    seed: int
    device: str
    paths: PathsConfig
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    train: TrainConfig
    tune: TuneConfig
    serve: ServeConfig


def load_config(path: str | Path = "params.yaml") -> Config:
    """Read params.yaml from the current directory (run commands from the project root)."""
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
