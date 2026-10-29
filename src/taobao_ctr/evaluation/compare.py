"""Results table assembly and variant selection (notebook section 11): for every model, the
variant (default or tuned) with the higher validation AUC is selected; its test-day metrics are
reported. Each training stage writes its own `metrics/<key>.json` (a dvc metric); this module
turns a list of those row dicts into the same results table the notebook built from its
in-memory `RESULTS` dict.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULT_COLS = [
    "model", "variant", "val_AUC", "AUC", "LogLoss", "PR_AUC", "head2_AUC", "best_step", "train_min", "params_M",
]


def save_result(path: str | Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(row, indent=1, default=float))


def load_results(metrics_dir: str | Path) -> list[dict]:
    """Every `<name>/<variant>.json` row written by a training stage under metrics_dir."""
    return [json.loads(p.read_text()) for p in sorted(Path(metrics_dir).rglob("*.json"))]


def results_table(rows: list[dict]) -> pd.DataFrame:
    """For every model, mark the variant with the higher validation AUC as `selected`."""
    res = pd.DataFrame(rows)
    res["selected"] = res["val_AUC"] == res.groupby("model")["val_AUC"].transform("max")
    return res.reindex(columns=RESULT_COLS + ["selected", "hp"])


def selected_row(res: pd.DataFrame, model: str) -> pd.Series:
    """The selected (default or tuned) row for one model."""
    return res[(res["model"] == model) & res["selected"]].iloc[0]
