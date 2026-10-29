"""The remaining dvc stages: train_lightgbm, train_deep, tune, evaluate, combine, register,
monitor. Split out of pipeline.py so `python -m taobao_ctr.pipeline --help` doesn't need to
import torch/lightgbm/mlflow just to print usage.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from taobao_ctr.config import AppConfig
from taobao_ctr.evaluation.combine import evaluate_combinations
from taobao_ctr.evaluation.compare import load_results, results_table, save_result, selected_row
from taobao_ctr.models.registry import MODELS
from taobao_ctr.models.towers import ThreeTower
from taobao_ctr.pipeline import load_dataset
from taobao_ctr.tracking.mlflow_utils import already_registered, init_tracking, register_champion
from taobao_ctr.training.lightgbm_train import lgb_predict, train_lightgbm
from taobao_ctr.training.trainer import (
    HP_KEYS,
    default_hp,
    free_memory,
    get_device,
    metrics,
    predict,
    train_model,
)
from taobao_ctr.training.tune import hp_from_params, should_adopt_tuned
from taobao_ctr.training.tune import tune as tune_model
from taobao_ctr.utils import slug


def _with_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pred_file(cfg: AppConfig, name: str, variant: str) -> Path:
    return _with_parent(cfg.paths.artifact_dir / "preds" / name / f"{variant}.npz")


def _model_file(cfg: AppConfig, name: str, variant: str, ext: str) -> Path:
    return _with_parent(cfg.paths.artifact_dir / "models" / name / f"{variant}.{ext}")


def _metrics_file(cfg: AppConfig, name: str, variant: str) -> Path:
    return _with_parent(cfg.paths.artifact_dir / "metrics" / name / f"{variant}.json")


def stage_train_lightgbm(cfg: AppConfig) -> None:
    init_tracking(cfg.paths.mlflow_tracking_uri)
    dataset, _train, fit_rows, val_rows, test_rows = load_dataset(cfg)
    key, variant = "LightGBM", "default"
    t0 = time.time()
    lgb_cfg = cfg.params.train.lightgbm
    params = {
        "objective": "binary", "learning_rate": lgb_cfg.learning_rate, "num_leaves": lgb_cfg.num_leaves,
        "min_data_in_leaf": lgb_cfg.min_data_in_leaf, "feature_fraction": lgb_cfg.feature_fraction,
        "bagging_fraction": lgb_cfg.bagging_fraction, "bagging_freq": lgb_cfg.bagging_freq,
        "metric": "auc", "verbose": -1, "seed": cfg.params.seed,
    }
    with mlflow.start_run(run_name=f"{key} [{variant}]"):
        mlflow.log_params({"model": key, "variant": variant, **params})
        gbm = train_lightgbm(
            dataset, fit_rows, val_rows, params,
            num_boost_round=lgb_cfg.num_boost_round, early_stopping_rounds=lgb_cfg.early_stopping_rounds,
            log_evaluation_every=lgb_cfg.log_evaluation_every,
        )
        p_val, p_test = lgb_predict(gbm, dataset, val_rows), lgb_predict(gbm, dataset, test_rows)
        np.savez(_pred_file(cfg, key, variant), logit_val=p_val, logit_test=p_test)
        gbm.save_model(str(_model_file(cfg, key, variant, "txt")))
        from sklearn.metrics import roc_auc_score

        row = {
            "model": key, "variant": variant, "val_AUC": roc_auc_score(dataset.y[val_rows], p_val),
            "best_step": gbm.best_iteration, **metrics(dataset.y[test_rows], p_test),
            "train_min": (time.time() - t0) / 60, "hp": params,
        }
        mlflow.log_metrics({k: row[k] for k in ["val_AUC", "AUC", "LogLoss", "PR_AUC", "train_min"]})
        mlflow.log_artifact(str(_model_file(cfg, key, variant, "txt")))
    save_result(_metrics_file(cfg, key, variant), row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "hp"})


def _run_deep_model(cfg: AppConfig, name: str, hp: dict, variant: str) -> dict:
    init_tracking(cfg.paths.mlflow_tracking_uri)
    dataset, _train, fit_rows, val_rows, test_rows = load_dataset(cfg)
    device = get_device()
    key = f"{name} [{variant}]"
    t0 = time.time()
    with mlflow.start_run(run_name=key):
        mlflow.log_params({"model": name, "variant": variant, **{k: hp[k] for k in HP_KEYS}})
        model, best, curve = train_model(
            MODELS[name], hp, dataset, fit_rows, val_rows, hp["max_epochs"], hp["eval_every_rows"], hp["patience"],
            device, cfg.params.seed,
        )
        for step, auc in curve:
            mlflow.log_metric("val_auc", auc, step=step)
        p_val, p_test = predict(model, dataset, val_rows, device), predict(model, dataset, test_rows, device)
        np.savez(_pred_file(cfg, name, variant), **{f"{h}_val": v for h, v in p_val.items()}, **{f"{h}_test": v for h, v in p_test.items()})
        torch.save(model.state_dict(), _model_file(cfg, name, variant, "pt"))
        row = {
            "model": name, "variant": variant, "val_AUC": best["auc"], "best_step": best["step"],
            **metrics(dataset.y[test_rows], p_test["logit"]), "train_min": (time.time() - t0) / 60,
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6, "hp": {k: hp[k] for k in HP_KEYS},
        }
        if "cotrain_logit" in p_test:
            from sklearn.metrics import roc_auc_score

            row["head2_AUC"] = roc_auc_score(dataset.y[test_rows], p_test["cotrain_logit"])
        mlflow.log_metrics({
            k: row[k] for k in ["val_AUC", "AUC", "LogLoss", "PR_AUC", "train_min"] + (["head2_AUC"] if "head2_AUC" in row else [])
        })
    save_result(_metrics_file(cfg, name, variant), row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "hp"})
    del model
    free_memory(device)
    return row


def stage_train_deep(cfg: AppConfig, name: str) -> None:
    hp = default_hp(cfg.params)
    _run_deep_model(cfg, name, hp, "default")


def stage_tune(cfg: AppConfig, name: str) -> None:
    init_tracking(cfg.paths.mlflow_tracking_uri)
    dataset, _train, fit_rows, val_rows, _test = load_dataset(cfg)
    device, hp = get_device(), default_hp(cfg.params)
    tcfg = cfg.params.tune
    optuna_db = cfg.paths.artifact_dir / "optuna.db"
    study = tune_model(
        name, hp, HP_KEYS, dataset, fit_rows, val_rows, device, cfg.params.seed,
        storage=f"sqlite:///{optuna_db}", n_trials=tcfg.tune_trials, timeout_min=tcfg.tune_timeout_min,
        tune_rows=tcfg.tune_rows, tune_val_rows=tcfg.tune_val_rows,
    )
    if not should_adopt_tuned(study):
        print(f"{name}: the default setting won the search; nothing to refit")
        # dvc still needs this stage's declared output to exist, and downstream code needs a
        # way to tell "no tuned variant" apart from "variant trained but worse": an incomplete
        # row (no val_AUC) never wins results_table()'s per-model max, so it is never selected.
        save_result(_metrics_file(cfg, name, "tuned"), {"model": name, "variant": "tuned", "adopted": False})
        return
    tuned_hp = hp_from_params(hp, study.best_params, HP_KEYS)
    _run_deep_model(cfg, name, tuned_hp, "tuned")


def stage_evaluate(cfg: AppConfig) -> None:
    rows = load_results(cfg.paths.artifact_dir / "metrics")
    res = results_table(rows)
    out = cfg.paths.artifact_dir / "results.csv"
    display_cols = [c for c in res.columns if c != "hp"]
    res[display_cols].to_csv(out, index=False)
    print(res[display_cols].sort_values(["selected", "AUC"], ascending=False).round(4))


def stage_combine(cfg: AppConfig) -> None:
    init_tracking(cfg.paths.mlflow_tracking_uri)
    dataset, _train, _fit, val_rows, test_rows = load_dataset(cfg)
    rows = load_results(cfg.paths.artifact_dir / "metrics")
    res = results_table(rows)

    def preds(model_name: str):
        row = selected_row(res, model_name)
        return np.load(_pred_file(cfg, model_name, row["variant"]))

    p3, pg = preds("3-tower co-train"), preds("LightGBM")
    P = {
        "3-tower": (p3["logit_val"], p3["logit_test"]),
        "two-tower head": (p3["cotrain_logit_val"], p3["cotrain_logit_test"]),
        "LightGBM": (pg["logit_val"], pg["logit_test"]),
    }
    combos, sweep = evaluate_combinations(
        P, dataset.y[val_rows], dataset.y[test_rows], cfg.params.serve.cascade_top_frac, metrics,
    )
    combos.to_csv(cfg.paths.artifact_dir / "combinations.csv")
    with mlflow.start_run(run_name="combinations"):
        mlflow.log_metrics({slug(m)[:250]: v for m, v in combos["AUC"].items()})
        mlflow.log_artifact(str(cfg.paths.artifact_dir / "combinations.csv"))
    print(combos.round(4))


def stage_register(cfg: AppConfig) -> None:
    init_tracking(cfg.paths.mlflow_tracking_uri)
    rows = load_results(cfg.paths.artifact_dir / "metrics")
    res = results_table(rows)
    row = selected_row(res, "3-tower co-train")
    key = f"3-tower co-train [{row['variant']}]"
    registered_path = cfg.paths.artifact_dir / "registered.json"
    if already_registered(registered_path, key):
        print(f"{key} is already registered as taobao-ctr-3tower@champion")
        return

    dataset, *_ = load_dataset(cfg)
    hp = hp_from_params(default_hp(cfg.params), row["hp"], HP_KEYS)
    model = ThreeTower(hp, dataset.sizes, dataset.seq_len)
    weights_path = _model_file(cfg, "3-tower co-train", row["variant"], "pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    register_champion(
        model, key, row["hp"], {"AUC": row["AUC"], "LogLoss": row["LogLoss"], "PR_AUC": row["PR_AUC"]},
        row["val_AUC"], registered_path, fallback_weights_path=weights_path,
    )


def stage_monitor(cfg: AppConfig) -> None:
    from taobao_ctr.features.click_rates import TE_COLS
    from taobao_ctr.monitoring.report import (
        auc_per_day,
        calibration_by_decile,
        prediction_distribution,
        psi_drift_report,
        render_report,
        write_report,
    )

    init_tracking(cfg.paths.mlflow_tracking_uri)
    dataset, _train, fit_rows, _val, test_rows = load_dataset(cfg)
    rows = load_results(cfg.paths.artifact_dir / "metrics")
    res = results_table(rows)
    row = selected_row(res, "3-tower co-train")
    y_pred_test = np.load(_pred_file(cfg, "3-tower co-train", row["variant"]))["logit_test"]

    calib = calibration_by_decile(dataset.y[test_rows], y_pred_test)
    auc_day = auc_per_day(dataset.y[test_rows], y_pred_test, dataset.ts[test_rows])
    drift = psi_drift_report(dataset.TE[fit_rows], dataset.TE[test_rows], TE_COLS)
    dist = prediction_distribution(y_pred_test)
    markdown = render_report(calib, auc_day, drift, dist)

    out_path = Path("reports") / "monitoring.md"
    write_report(out_path, markdown)
    with mlflow.start_run(run_name="monitoring"):
        mlflow.log_metrics({f"psi_{r['feature']}": r["PSI"] for r in drift.to_dict("records")})
        mlflow.log_artifact(str(out_path))
    print(f"monitoring report written to {out_path}")


def run(stage: str, cfg: AppConfig, name: str | None) -> None:
    if stage == "train_lightgbm":
        stage_train_lightgbm(cfg)
    elif stage == "train_deep":
        if not name:
            raise SystemExit("train_deep needs --name <model name>")
        stage_train_deep(cfg, name)
    elif stage == "tune":
        if not name:
            raise SystemExit("tune needs --name <model name>")
        stage_tune(cfg, name)
    elif stage == "evaluate":
        stage_evaluate(cfg)
    elif stage == "combine":
        stage_combine(cfg)
    elif stage == "register":
        stage_register(cfg)
    elif stage == "monitor":
        stage_monitor(cfg)
    else:
        from taobao_ctr import pipeline

        {
            "prepare_impressions": pipeline.stage_prepare_impressions,
            "build_history": pipeline.stage_build_history,
            "click_rates": pipeline.stage_click_rates,
        }[stage](cfg)
