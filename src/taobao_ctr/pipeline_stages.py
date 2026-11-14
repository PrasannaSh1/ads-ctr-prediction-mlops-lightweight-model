"""The 7 model stages: train_lightgbm, train_deep, tune, evaluate, combine, register, monitor.

They live in their own file so that `python -m taobao_ctr.pipeline --help` does not need to
import torch, lightgbm and mlflow first.
"""
import os
import time

import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from taobao_ctr.evaluation.combine import evaluate_combinations
from taobao_ctr.evaluation.compare import load_results, results_table, save_result, selected_row
from taobao_ctr.features.click_rates import TE_COLS
from taobao_ctr.models.registry import MODELS
from taobao_ctr.models.towers import ThreeTower
from taobao_ctr.monitoring.report import (
    auc_per_day,
    calibration_by_decile,
    prediction_distribution,
    psi_drift_report,
    render_report,
    write_report,
)
from taobao_ctr.pipeline import load_dataset, stage_build_history, stage_click_rates, stage_prepare_impressions
from taobao_ctr.tracking.mlflow_utils import already_registered, init_tracking, register_champion
from taobao_ctr.training.lightgbm_train import lgb_predict, train_lightgbm
from taobao_ctr.training.trainer import HP_KEYS, default_hp, free_memory, metrics, predict, train_model
from taobao_ctr.training.tune import hp_from_params, should_adopt_tuned, tune
from taobao_ctr.utils import slug


# ---------------------------------------------------------------------------
# Where the output files go
# ---------------------------------------------------------------------------

def output_path(config, kind, name, file_name):
    """artifacts/<kind>/<model name>/<file name>, creating the folder if needed."""
    folder = os.path.join(config["paths"]["artifact_dir"], kind, name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, file_name)


def pred_file(config, name, variant):
    return output_path(config, "preds", name, variant + ".npz")


def model_file(config, name, variant, extension):
    return output_path(config, "models", name, variant + "." + extension)


def metrics_file(config, name, variant):
    return output_path(config, "metrics", name, variant + ".json")


def print_row(row):
    """Print a metrics row with the numbers rounded, without the long hp dict."""
    shown = {}
    for key, value in row.items():
        if key == "hp":
            continue
        if isinstance(value, float):
            value = round(value, 4)
        shown[key] = value
    print(shown)


def main_metrics(row):
    """The metrics every training run logs to MLflow."""
    logged = {}
    for key in ["val_AUC", "AUC", "LogLoss", "PR_AUC", "train_min"]:
        logged[key] = row[key]
    if "head2_AUC" in row:
        logged["head2_AUC"] = row["head2_AUC"]
    return logged


# ---------------------------------------------------------------------------
# Stage 4: train_lightgbm
# ---------------------------------------------------------------------------

def stage_train_lightgbm(config):
    init_tracking(config["paths"]["mlflow_tracking_uri"])
    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    name = "LightGBM"
    variant = "default"
    start_time = time.time()

    lgb_config = config["train"]["lightgbm"]
    params = {
        "objective": "binary",
        "learning_rate": lgb_config["learning_rate"],
        "num_leaves": lgb_config["num_leaves"],
        "min_data_in_leaf": lgb_config["min_data_in_leaf"],
        "feature_fraction": lgb_config["feature_fraction"],
        "bagging_fraction": lgb_config["bagging_fraction"],
        "bagging_freq": lgb_config["bagging_freq"],
        "metric": "auc",
        "verbose": -1,
        "seed": config["seed"],
    }

    with mlflow.start_run(run_name=f"{name} [{variant}]"):
        logged_params = {"model": name, "variant": variant}
        logged_params.update(params)
        mlflow.log_params(logged_params)

        gbm = train_lightgbm(
            dataset, fit_rows, val_rows, params,
            num_boost_round=lgb_config["num_boost_round"],
            early_stopping_rounds=lgb_config["early_stopping_rounds"],
            log_evaluation_every=lgb_config["log_evaluation_every"],
        )
        p_val = lgb_predict(gbm, dataset, val_rows)
        p_test = lgb_predict(gbm, dataset, test_rows)
        np.savez(pred_file(config, name, variant), logit_val=p_val, logit_test=p_test)
        gbm.save_model(model_file(config, name, variant, "txt"))

        row = {
            "model": name,
            "variant": variant,
            "val_AUC": roc_auc_score(dataset.y[val_rows], p_val),
            "best_step": gbm.best_iteration,
        }
        row.update(metrics(dataset.y[test_rows], p_test))
        row["train_min"] = (time.time() - start_time) / 60
        row["hp"] = params
        mlflow.log_metrics(main_metrics(row))
        mlflow.log_artifact(model_file(config, name, variant, "txt"))

    save_result(metrics_file(config, name, variant), row)
    print_row(row)


# ---------------------------------------------------------------------------
# Stage 5: train_deep (one deep model with the default settings)
# Stage 6: tune (Optuna search, then retrain with the best settings if they won)
# ---------------------------------------------------------------------------

def run_deep_model(config, name, hp, variant):
    """Train one deep model, save its predictions, weights and metrics, log everything to MLflow."""
    init_tracking(config["paths"]["mlflow_tracking_uri"])
    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    device = torch.device(config["device"])
    start_time = time.time()

    with mlflow.start_run(run_name=f"{name} [{variant}]"):
        logged_params = {"model": name, "variant": variant}
        for key in HP_KEYS:
            logged_params[key] = hp[key]
        mlflow.log_params(logged_params)

        model, best, curve = train_model(
            MODELS[name], hp, dataset, fit_rows, val_rows,
            hp["max_epochs"], hp["eval_every_rows"], hp["patience"], device, config["seed"],
        )
        for step, auc in curve:
            mlflow.log_metric("val_auc", auc, step=step)

        p_val = predict(model, dataset, val_rows, device)
        p_test = predict(model, dataset, test_rows, device)
        saved_predictions = {}
        for head_name in p_val:
            saved_predictions[head_name + "_val"] = p_val[head_name]
        for head_name in p_test:
            saved_predictions[head_name + "_test"] = p_test[head_name]
        np.savez(pred_file(config, name, variant), **saved_predictions)
        torch.save(model.state_dict(), model_file(config, name, variant, "pt"))

        n_params = 0
        for p in model.parameters():
            n_params += p.numel()
        used_hp = {}
        for key in HP_KEYS:
            used_hp[key] = hp[key]

        row = {
            "model": name,
            "variant": variant,
            "val_AUC": best["auc"],
            "best_step": best["step"],
        }
        row.update(metrics(dataset.y[test_rows], p_test["logit"]))
        row["train_min"] = (time.time() - start_time) / 60
        row["params_M"] = n_params / 1e6
        row["hp"] = used_hp
        if "cotrain_logit" in p_test:
            row["head2_AUC"] = roc_auc_score(dataset.y[test_rows], p_test["cotrain_logit"])
        mlflow.log_metrics(main_metrics(row))

    save_result(metrics_file(config, name, variant), row)
    print_row(row)
    del model
    free_memory()
    return row


def stage_train_deep(config, name):
    hp = default_hp(config)
    run_deep_model(config, name, hp, "default")


def stage_tune(config, name):
    init_tracking(config["paths"]["mlflow_tracking_uri"])
    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    device = torch.device(config["device"])
    hp = default_hp(config)
    tune_config = config["tune"]
    optuna_db = os.path.join(config["paths"]["artifact_dir"], "optuna.db")

    study = tune(
        name, hp, HP_KEYS, dataset, fit_rows, val_rows, device, config["seed"],
        storage="sqlite:///" + optuna_db,
        n_trials=tune_config["tune_trials"],
        timeout_min=tune_config["tune_timeout_min"],
        tune_rows=tune_config["tune_rows"],
        tune_val_rows=tune_config["tune_val_rows"],
    )

    if not should_adopt_tuned(study):
        print(f"{name}: the default setting won the search; nothing to refit")
        # DVC still expects this stage's output file. A row without val_AUC can never be
        # selected by results_table(), so it does not change the results.
        save_result(metrics_file(config, name, "tuned"), {"model": name, "variant": "tuned", "adopted": False})
        return

    tuned_hp = hp_from_params(hp, study.best_params, HP_KEYS)
    run_deep_model(config, name, tuned_hp, "tuned")


# ---------------------------------------------------------------------------
# Stage 7: evaluate (the results table)
# ---------------------------------------------------------------------------

def stage_evaluate(config):
    rows = load_results(os.path.join(config["paths"]["artifact_dir"], "metrics"))
    res = results_table(rows)
    columns_to_show = []
    for col in res.columns:
        if col != "hp":
            columns_to_show.append(col)
    res[columns_to_show].to_csv(os.path.join(config["paths"]["artifact_dir"], "results.csv"), index=False)
    print(res[columns_to_show].sort_values(["selected", "AUC"], ascending=False).round(4))


# ---------------------------------------------------------------------------
# Stage 8: combine (cascades and blends)
# ---------------------------------------------------------------------------

def load_predictions(config, res, model_name):
    """The saved predictions of a model's selected variant."""
    row = selected_row(res, model_name)
    return np.load(pred_file(config, model_name, row["variant"]))


def stage_combine(config):
    init_tracking(config["paths"]["mlflow_tracking_uri"])
    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    rows = load_results(os.path.join(config["paths"]["artifact_dir"], "metrics"))
    res = results_table(rows)

    p3 = load_predictions(config, res, "3-tower co-train")
    pg = load_predictions(config, res, "LightGBM")
    P = {
        "3-tower": (p3["logit_val"], p3["logit_test"]),
        "two-tower head": (p3["cotrain_logit_val"], p3["cotrain_logit_test"]),
        "LightGBM": (pg["logit_val"], pg["logit_test"]),
    }
    combos, sweep = evaluate_combinations(
        P, dataset.y[val_rows], dataset.y[test_rows], config["serve"]["cascade_top_frac"], metrics,
    )

    combos_path = os.path.join(config["paths"]["artifact_dir"], "combinations.csv")
    combos.to_csv(combos_path)
    with mlflow.start_run(run_name="combinations"):
        logged = {}
        for method, auc in combos["AUC"].items():
            logged[slug(method)[:250]] = auc
        mlflow.log_metrics(logged)
        mlflow.log_artifact(combos_path)
    print(combos.round(4))


# ---------------------------------------------------------------------------
# Stage 9: register (the champion model in the MLflow model registry)
# ---------------------------------------------------------------------------

def stage_register(config):
    init_tracking(config["paths"]["mlflow_tracking_uri"])
    rows = load_results(os.path.join(config["paths"]["artifact_dir"], "metrics"))
    res = results_table(rows)
    row = selected_row(res, "3-tower co-train")
    key = f"3-tower co-train [{row['variant']}]"
    registered_path = os.path.join(config["paths"]["artifact_dir"], "registered.json")
    if already_registered(registered_path, key):
        print(f"{key} is already registered as taobao-ctr-3tower@champion")
        return

    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    hp = hp_from_params(default_hp(config), row["hp"], HP_KEYS)
    model = ThreeTower(hp, dataset.sizes, dataset.seq_len)
    weights_path = model_file(config, "3-tower co-train", row["variant"], "pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    test_metrics = {"AUC": row["AUC"], "LogLoss": row["LogLoss"], "PR_AUC": row["PR_AUC"]}
    register_champion(model, key, row["hp"], test_metrics, row["val_AUC"], registered_path,
                      fallback_weights_path=weights_path)


# ---------------------------------------------------------------------------
# Stage 10: monitor (calibration, AUC per day, drift)
# ---------------------------------------------------------------------------

def stage_monitor(config):
    init_tracking(config["paths"]["mlflow_tracking_uri"])
    dataset, train_rows, fit_rows, val_rows, test_rows = load_dataset(config)
    rows = load_results(os.path.join(config["paths"]["artifact_dir"], "metrics"))
    res = results_table(rows)
    row = selected_row(res, "3-tower co-train")
    y_pred_test = np.load(pred_file(config, "3-tower co-train", row["variant"]))["logit_test"]

    calibration = calibration_by_decile(dataset.y[test_rows], y_pred_test)
    auc_by_day = auc_per_day(dataset.y[test_rows], y_pred_test, dataset.ts[test_rows])
    drift = psi_drift_report(dataset.TE[fit_rows], dataset.TE[test_rows], TE_COLS)
    distribution = prediction_distribution(y_pred_test)
    markdown = render_report(calibration, auc_by_day, drift, distribution)

    report_path = os.path.join("reports", "monitoring.md")
    write_report(report_path, markdown)
    with mlflow.start_run(run_name="monitoring"):
        logged = {}
        for r in drift.to_dict("records"):
            logged["psi_" + r["feature"]] = r["PSI"]
        mlflow.log_metrics(logged)
        mlflow.log_artifact(report_path)
    print(f"monitoring report written to {report_path}")


# ---------------------------------------------------------------------------
# Run the stage that was asked for
# ---------------------------------------------------------------------------

def run(stage, config, name):
    if stage == "prepare_impressions":
        stage_prepare_impressions(config)
    elif stage == "build_history":
        stage_build_history(config)
    elif stage == "click_rates":
        stage_click_rates(config)
    elif stage == "train_lightgbm":
        stage_train_lightgbm(config)
    elif stage == "train_deep":
        if not name:
            raise SystemExit("train_deep needs --name <model name>")
        stage_train_deep(config, name)
    elif stage == "tune":
        if not name:
            raise SystemExit("tune needs --name <model name>")
        stage_tune(config, name)
    elif stage == "evaluate":
        stage_evaluate(config)
    elif stage == "combine":
        stage_combine(config)
    elif stage == "register":
        stage_register(config)
    elif stage == "monitor":
        stage_monitor(config)
    else:
        raise SystemExit("unknown stage: " + str(stage))
