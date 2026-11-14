"""MODELS: the friendly name of each deep model -> its class."""
from taobao_ctr.models.baselines import MLP, MLPSequence, Transformer
from taobao_ctr.models.towers import ThreeTower, TwoTower

MODELS = {
    "3-tower co-train": ThreeTower,
    "Two-tower": TwoTower,
    "MLP": MLP,
    "MLP + sequence": MLPSequence,
    "Transformer": Transformer,
}
