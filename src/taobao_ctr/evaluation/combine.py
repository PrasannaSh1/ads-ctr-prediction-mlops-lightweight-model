"""Cascades (two-tower head -> 3-tower on the top cascade_top_frac, LightGBM -> 3-tower) and
logit-space blending (notebook section 12).

How Pinterest serves the 3-tower (engineering blog, "Beyond Two Towers", Part 2): a cheap
stage-1 score ranks all candidates, the 3-tower re-scores only the top K%, the rest keep the
stage-1 score. Taobao has no candidate lists per request, so offline this shows how much
accuracy the cheap first stage costs, rather than a real latency win (that is
scripts/benchmark_latency.py's job, Phase 5).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def cascade(stage1: np.ndarray, stage2: np.ndarray, frac: float) -> np.ndarray:
    """stage1's score everywhere, except the top `frac` by stage1 score, which get stage2's."""
    out = stage1.copy()
    top = np.argsort(-stage1)[: int(frac * len(stage1))]
    out[top] = stage2[top]
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def blend(pa: np.ndarray, pb: np.ndarray, w: float) -> np.ndarray:
    """Weighted average of two probabilities in logit space."""
    return 1 / (1 + np.exp(-(w * _logit(pa) + (1 - w) * _logit(pb))))


def evaluate_combinations(
    P: dict[str, tuple[np.ndarray, np.ndarray]],  # name -> (val_scores, test_scores), probability space
    y_val: np.ndarray,
    y_test: np.ndarray,
    cascade_top_frac: float,
    metrics_fn: Callable[[np.ndarray, np.ndarray], dict],
    fracs: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8),
    weights: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (combos, sweep): combos has one row per method tried, sweep has the cascade
    top_frac sweep used to pick the validation-best fraction."""
    weights = np.linspace(0, 1, 21) if weights is None else np.asarray(weights)
    rows = [{"method": f"{m} alone", **metrics_fn(y_test, P[m][1])} for m in P]
    sweep = []
    for stage1 in ["two-tower head", "LightGBM"]:
        for frac in fracs:
            sweep.append({
                "stage 1": stage1, "top_frac": frac,
                "val_AUC": roc_auc_score(y_val, cascade(P[stage1][0], P["3-tower"][0], frac)),
                "test_AUC": roc_auc_score(y_test, cascade(P[stage1][1], P["3-tower"][1], frac)),
            })
        k = cascade_top_frac
        rows.append({
            "method": f"{stage1} -> 3-tower on top {k:.0%}",
            **metrics_fn(y_test, cascade(P[stage1][1], P["3-tower"][1], k)),
        })
        s = pd.DataFrame(sweep)
        kb = s[s["stage 1"] == stage1].sort_values("val_AUC").iloc[-1]["top_frac"]
        rows.append({
            "method": f"{stage1} -> 3-tower on top {kb:.0%} (val-chosen)",
            **metrics_fn(y_test, cascade(P[stage1][1], P["3-tower"][1], kb)),
        })
    wb = weights[np.argmax([roc_auc_score(y_val, blend(P["LightGBM"][0], P["3-tower"][0], w)) for w in weights])]
    rows.append({
        "method": f"blend {wb:.2f} x LightGBM + {1 - wb:.2f} x 3-tower (val-chosen)",
        **metrics_fn(y_test, blend(P["LightGBM"][1], P["3-tower"][1], wb)),
    })
    combos = pd.DataFrame(rows).set_index("method").sort_values("AUC", ascending=False)
    return combos, pd.DataFrame(sweep)
