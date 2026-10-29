"""BCE loss (optionally weighted or focal) (notebook section 7). The 3-tower's combined loss
(BCE(rich) + cotrain_weight * BCE(two-tower head)) is assembled in trainer.train_model(), which
is the only place that knows a model has a second ("cotrain_logit") head."""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def make_loss(hp: dict, device: torch.device) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if hp["loss"] == "weighted_bce":  # clicks count pos_weight times more
        w = torch.tensor(hp["pos_weight"], device=device)
        return lambda z, t: F.binary_cross_entropy_with_logits(z, t, pos_weight=w)
    if hp["loss"] == "focal":  # down-weights easy examples
        def focal(z, t):
            ce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
            p = torch.sigmoid(z)
            return ((1 - (p * t + (1 - p) * (1 - t))) ** hp["gamma"] * ce).mean()
        return focal
    return lambda z, t: F.binary_cross_entropy_with_logits(z, t)
