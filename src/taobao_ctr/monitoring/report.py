"""The monitoring report: calibration by decile, AUC per day, feature drift (PSI) and the
distribution of the predictions. Written as a markdown file and logged to MLflow."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PSI_FEATURE_NOTE = "PSI < 0.1: no significant shift. 0.1-0.25: moderate. > 0.25: significant drift, consider retraining."


def calibration_by_decile(y_true, y_pred, n_bins=10):
    """Sort the rows by predicted CTR into n_bins equal groups; compare predicted vs actual CTR."""
    order = np.argsort(y_pred)
    groups = np.array_split(order, n_bins)
    rows = []
    for i, group in enumerate(groups):
        if len(group) == 0:
            continue
        rows.append({
            "decile": i + 1,
            "n": len(group),
            "predicted_ctr": float(y_pred[group].mean()),
            "actual_ctr": float(y_true[group].mean()),
        })
    return pd.DataFrame(rows)


def auc_per_day(y_true, y_pred, ts, day_seconds=86400):
    day = (ts - ts.min()) // day_seconds
    rows = []
    for d in sorted(np.unique(day)):
        mask = day == d
        labels = y_true[mask]
        if labels.min() != labels.max():
            auc = roc_auc_score(labels, y_pred[mask])
        else:
            auc = float("nan")  # AUC needs both clicks and non-clicks
        rows.append({"day": int(d), "n": int(mask.sum()), "AUC": auc})
    return pd.DataFrame(rows)


def psi(expected, actual, n_bins=10):
    """Population Stability Index: how much a feature's distribution changed between two samples."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_share = np.clip(expected_counts / len(expected), 1e-6, None)
    actual_share = np.clip(actual_counts / len(actual), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_drift_report(TE_train, TE_test, te_cols):
    rows = []
    for i, name in enumerate(te_cols):
        rows.append({"feature": name, "PSI": psi(TE_train[:, i], TE_test[:, i])})
    return pd.DataFrame(rows)


def prediction_distribution(y_pred, n_bins=20):
    counts, edges = np.histogram(y_pred.astype(np.float64), bins=n_bins, range=(0, 1))
    return pd.DataFrame({"bin_start": edges[:-1].round(3), "bin_end": edges[1:].round(3), "count": counts})


def table_to_markdown(df):
    """A pandas DataFrame as a markdown table."""
    header = "| " + " | ".join(df.columns) + " |"
    separator_cells = []
    for _ in df.columns:
        separator_cells.append("---")
    separator = "| " + " | ".join(separator_cells) + " |"
    lines = [header, separator]
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(calibration, auc_by_day, drift, pred_dist):
    parts = [
        "# Monitoring report\n",
        "## Calibration by predicted-CTR decile\n",
        table_to_markdown(calibration.round(4)),
        "\n\n## AUC per day\n",
        table_to_markdown(auc_by_day.round(4)),
        f"\n\n## Click-rate feature drift (PSI, test vs. fit)\n\n{PSI_FEATURE_NOTE}\n",
        table_to_markdown(drift.round(4)),
        "\n\n## Prediction distribution\n",
        table_to_markdown(pred_dist),
    ]
    return "\n".join(parts)


def write_report(path, markdown):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(markdown)
