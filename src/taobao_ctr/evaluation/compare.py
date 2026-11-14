"""Builds the results table. Every training stage writes one small JSON file with its metrics;
this module reads them all and, for every model, picks the variant (default or tuned) with the
higher validation AUC."""
import json
import os
from pathlib import Path

import pandas as pd

RESULT_COLS = [
    "model", "variant", "val_AUC", "AUC", "LogLoss", "PR_AUC", "head2_AUC", "best_step", "train_min", "params_M",
]


def save_result(path, row):
    """Write one metrics row as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(row, f, indent=1, default=float)


def load_results(metrics_dir):
    """Read every <model>/<variant>.json file under metrics_dir. Returns a list of dicts."""
    rows = []
    for path in sorted(Path(metrics_dir).rglob("*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def results_table(rows):
    """One row per (model, variant). The "selected" column marks the variant with the higher val AUC."""
    res = pd.DataFrame(rows)
    best_per_model = res.groupby("model")["val_AUC"].transform("max")
    res["selected"] = res["val_AUC"] == best_per_model
    return res.reindex(columns=RESULT_COLS + ["selected", "hp"])


def selected_row(res, model):
    """The selected (default or tuned) row of one model."""
    is_this_model = res["model"] == model
    return res[is_this_model & res["selected"]].iloc[0]
