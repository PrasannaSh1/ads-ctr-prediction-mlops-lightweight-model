"""Combinations of models, checked offline:

- Cascade: a cheap stage-1 score ranks all candidates, the 3-tower re-scores only the top K%,
  the rest keep the stage-1 score. Shows how much accuracy the cheap first stage costs.
- Blend: a weighted average of LightGBM and the 3-tower, in logit space.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def cascade(stage1, stage2, frac):
    """stage1's scores everywhere, except the top `frac` of rows (by stage1 score) get stage2's."""
    out = stage1.copy()
    n_top = int(frac * len(stage1))
    top = np.argsort(-stage1)[:n_top]
    out[top] = stage2[top]
    return out


def to_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def blend(pa, pb, w):
    """Weighted average of two probabilities, done in logit space."""
    mixed_logit = w * to_logit(pa) + (1 - w) * to_logit(pb)
    return 1 / (1 + np.exp(-mixed_logit))


def evaluate_combinations(P, y_val, y_test, cascade_top_frac, metrics_fn, fracs=(0.1, 0.2, 0.4, 0.6, 0.8),
                          weights=None):
    """Try every combination and score it on the test day.

    P is a dict: model name -> (validation scores, test scores).
    Returns (combos, sweep): combos has one row per method, sweep has every cascade fraction tried.
    """
    if weights is None:
        weights = np.linspace(0, 1, 21)
    else:
        weights = np.asarray(weights)

    rows = []

    # Each model alone
    for name in P:
        row = {"method": name + " alone"}
        row.update(metrics_fn(y_test, P[name][1]))
        rows.append(row)

    # Cascades: stage 1 = the two-tower head, or LightGBM; stage 2 = the 3-tower
    sweep = []
    for stage1 in ["two-tower head", "LightGBM"]:
        stage1_val = P[stage1][0]
        stage1_test = P[stage1][1]
        tower_val = P["3-tower"][0]
        tower_test = P["3-tower"][1]

        # Try every fraction, on validation and test
        for frac in fracs:
            val_auc = roc_auc_score(y_val, cascade(stage1_val, tower_val, frac))
            test_auc = roc_auc_score(y_test, cascade(stage1_test, tower_test, frac))
            sweep.append({"stage 1": stage1, "top_frac": frac, "val_AUC": val_auc, "test_AUC": test_auc})

        # The fraction from params.yaml
        k = cascade_top_frac
        row = {"method": f"{stage1} -> 3-tower on top {k:.0%}"}
        row.update(metrics_fn(y_test, cascade(stage1_test, tower_test, k)))
        rows.append(row)

        # The fraction that did best on validation
        sweep_df = pd.DataFrame(sweep)
        this_stage = sweep_df[sweep_df["stage 1"] == stage1]
        best_frac = this_stage.sort_values("val_AUC").iloc[-1]["top_frac"]
        row = {"method": f"{stage1} -> 3-tower on top {best_frac:.0%} (val-chosen)"}
        row.update(metrics_fn(y_test, cascade(stage1_test, tower_test, best_frac)))
        rows.append(row)

    # Blend: pick the weight that does best on validation
    val_aucs = []
    for w in weights:
        val_aucs.append(roc_auc_score(y_val, blend(P["LightGBM"][0], P["3-tower"][0], w)))
    best_w = weights[np.argmax(val_aucs)]
    row = {"method": f"blend {best_w:.2f} x LightGBM + {1 - best_w:.2f} x 3-tower (val-chosen)"}
    row.update(metrics_fn(y_test, blend(P["LightGBM"][1], P["3-tower"][1], best_w)))
    rows.append(row)

    combos = pd.DataFrame(rows).set_index("method").sort_values("AUC", ascending=False)
    return combos, pd.DataFrame(sweep)
