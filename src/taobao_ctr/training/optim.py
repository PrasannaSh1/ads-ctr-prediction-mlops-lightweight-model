"""Optimizer construction from hp (notebook section 7)."""
from __future__ import annotations

import torch


def make_optimizer(model: torch.nn.Module, hp: dict) -> torch.optim.Optimizer:
    opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "adagrad": torch.optim.Adagrad}[hp["optimizer"]]
    return opt(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
