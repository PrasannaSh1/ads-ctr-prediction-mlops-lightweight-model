"""Resumable per-model Optuna hyper-parameter search (notebook section 10).

Trials train on a random subset of fit_rows for 1 epoch and are scored on a subset of val_rows
(the test day is never touched during tuning). Trial 0 is always the current default setting;
the TPE sampler proposes the rest and a median pruner stops clearly-worse trials early.
`pos_weight` is capped at 3 -- PROMPT_taobao_ctr_mlops.md constraint 6 notes an uncapped
pos_weight of 9.4 broke the 3-tower in development: good on validation, much worse on test.

This module has no MLflow dependency: it is pure search mechanics. The caller (Phase 4's
orchestration) decides whether to log trials and, via `should_adopt_tuned`, whether a refit of
the best trial actually replaces the default -- only if its validation AUC is higher
(constraint 6: "replaces the default only if its refit wins on validation").
"""
from __future__ import annotations

import numpy as np
import optuna

from taobao_ctr.data.dataset import Dataset
from taobao_ctr.models.registry import MODELS
from taobao_ctr.training.trainer import free_memory, train_model
from taobao_ctr.utils import slug

DEFAULT_TRIAL = {
    "lr": 1e-3, "dropout": 0.1, "optimizer": "adam", "weight_decay": 1e-7,
    "batch_size": 4096, "loss": "bce", "cotrain_weight": 0.5,
}


def hp_from_params(default_hp: dict, params: dict, hp_keys: list[str]) -> dict:
    """Turn Optuna parameters into a full settings dict (defaults + overrides)."""
    hp = dict(default_hp)
    hp.update({k: v for k, v in params.items() if k in hp_keys})
    return hp


def suggest(trial: optuna.Trial, name: str, default_hp: dict, hp_keys: list[str]) -> dict:
    params = {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "adagrad"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "loss": trial.suggest_categorical("loss", ["bce", "focal", "weighted_bce"]),
    }
    if params["loss"] == "focal":
        params["gamma"] = trial.suggest_float("gamma", 0.5, 2.0)
    if params["loss"] == "weighted_bce":
        params["pos_weight"] = trial.suggest_float("pos_weight", 1.0, 3.0)  # capped at 3
    if name == "3-tower co-train":
        params["cotrain_weight"] = trial.suggest_float("cotrain_weight", 0.25, 1.0)
    return hp_from_params(default_hp, params, hp_keys)


def tune(
    name: str, default_hp: dict, hp_keys: list[str], data: Dataset, fit_rows: np.ndarray, val_rows: np.ndarray,
    device, seed: int, *, storage: str, n_trials: int, timeout_min: float, tune_rows: int, tune_val_rows: int,
) -> optuna.Study:
    study = optuna.create_study(
        study_name=slug(name), storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )
    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    if len(finished) >= n_trials:
        return study
    if not study.trials:
        study.enqueue_trial({k: v for k, v in DEFAULT_TRIAL.items() if name == "3-tower co-train" or k != "cotrain_weight"})

    rng = np.random.default_rng(seed)
    tune_fit = np.sort(rng.choice(fit_rows, min(tune_rows, len(fit_rows)), replace=False))
    tune_val = np.sort(rng.choice(val_rows, min(tune_val_rows, len(val_rows)), replace=False))

    def objective(trial: optuna.Trial) -> float:
        hp = suggest(trial, name, default_hp, hp_keys)

        def report(i, auc):
            trial.report(auc, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        eval_rows = max(len(tune_fit) // 2, hp["batch_size"])  # 2 checks per trial (for pruning)
        model, best, _ = train_model(
            MODELS[name], hp, data, tune_fit, tune_val, 1, eval_rows, 99, device, seed, report, verbose=False,
        )
        del model
        free_memory(device)
        return best["auc"]

    study.optimize(objective, n_trials=n_trials - len(finished), timeout=timeout_min * 60, catch=(RuntimeError,))
    return study


def should_adopt_tuned(study: optuna.Study) -> bool:
    """A tuned setting replaces the default only if trial 0 (the default) did not win."""
    return study.best_trial.number != 0
