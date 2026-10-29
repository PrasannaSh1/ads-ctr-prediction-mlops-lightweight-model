"""MODELS: friendly name -> model class, used by training/tuning/evaluation to iterate over all
5 deep models by name (notebook section 6/9)."""
from __future__ import annotations

from taobao_ctr.models.baselines import MLP, MLPSequence, Transformer
from taobao_ctr.models.towers import ThreeTower, TwoTower

MODELS = {
    "3-tower co-train": ThreeTower,
    "Two-tower": TwoTower,
    "MLP": MLP,
    "MLP + sequence": MLPSequence,
    "Transformer": Transformer,
}
