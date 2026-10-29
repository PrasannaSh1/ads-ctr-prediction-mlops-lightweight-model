"""Offline monitoring report: calibration by predicted-CTR decile, AUC per day, PSI drift of
each click-rate feature (test vs. training), and the prediction distribution. Written as
markdown (no extra templating dependency) and logged to MLflow as an artifact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_FEATURE_NOTE = "PSI < 0.1: no significant shift. 0.1-0.25: moderate. > 0.25: significant drift, consider retraining."


def calibration_by_decile(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(y_pred)
    bins = np.array_split(order, n_bins)
    rows = [
        {"decile": i + 1, "n": len(b), "predicted_ctr": float(y_pred[b].mean()), "actual_ctr": float(y_true[b].mean())}
        for i, b in enumerate(bins) if len(b) > 0
    ]
    return pd.DataFrame(rows)


def auc_per_day(y_true: np.ndarray, y_pred: np.ndarray, ts: np.ndarray, day_seconds: int = 86_400) -> pd.DataFrame:
    day = (ts - ts.min()) // day_seconds
    rows = []
    for d in sorted(np.unique(day)):
        mask = day == d
        yt = y_true[mask]
        auc = roc_auc_score(yt, y_pred[mask]) if yt.min() != yt.max() else float("nan")
        rows.append({"day": int(d), "n": int(mask.sum()), "AUC": auc})
    return pd.DataFrame(rows)


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two samples of the same feature."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / len(expected), 1e-6, None)
    act_pct = np.clip(act_counts / len(actual), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_drift_report(TE_train: np.ndarray, TE_test: np.ndarray, te_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"feature": c, "PSI": psi(TE_train[:, i], TE_test[:, i])} for i, c in enumerate(te_cols)])


def prediction_distribution(y_pred: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    counts, edges = np.histogram(y_pred.astype(np.float64), bins=n_bins, range=(0, 1))
    return pd.DataFrame({"bin_start": edges[:-1].round(3), "bin_end": edges[1:].round(3), "count": counts})


def _df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def render_report(
    calibration: pd.DataFrame, auc_by_day: pd.DataFrame, drift: pd.DataFrame, pred_dist: pd.DataFrame,
) -> str:
    parts = [
        "# Monitoring report\n",
        "## Calibration by predicted-CTR decile\n",
        _df_to_markdown(calibration.round(4)),
        "\n\n## AUC per day\n",
        _df_to_markdown(auc_by_day.round(4)),
        f"\n\n## Click-rate feature drift (PSI, test vs. fit)\n\n{PSI_FEATURE_NOTE}\n",
        _df_to_markdown(drift.round(4)),
        "\n\n## Prediction distribution\n",
        _df_to_markdown(pred_dist),
    ]
    return "\n".join(parts)


def write_report(path: str | Path, markdown: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(markdown)
