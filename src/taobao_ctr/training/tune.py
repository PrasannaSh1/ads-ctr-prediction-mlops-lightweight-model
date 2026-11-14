"""Hyper-parameter search with Optuna, one study per deep model.

Every trial trains for 1 epoch on a random part of the fit rows and is scored on a random part
of the validation rows (the test day is never used). Trial 0 is always the default setting.
The study is saved in a SQLite file, so a stopped search continues where it left off.

pos_weight is capped at 3: an uncapped value of 9.4 broke the 3-tower in development
(good on validation, much worse on test).

A tuned setting replaces the default only if the best trial is not trial 0 (should_adopt_tuned).
"""
import numpy as np
import optuna

from taobao_ctr.models.registry import MODELS
from taobao_ctr.training.trainer import free_memory, train_model
from taobao_ctr.utils import slug

# Trial 0: the default settings
DEFAULT_TRIAL = {
    "lr": 1e-3,
    "dropout": 0.1,
    "optimizer": "adam",
    "weight_decay": 1e-7,
    "batch_size": 4096,
    "loss": "bce",
    "cotrain_weight": 0.5,
}


def hp_from_params(default_hp, params, hp_keys):
    """Start from the default settings and overwrite the ones Optuna chose."""
    hp = dict(default_hp)
    for key, value in params.items():
        if key in hp_keys:
            hp[key] = value
    return hp


def suggest(trial, name, default_hp, hp_keys):
    """Ask Optuna for one set of settings to try."""
    params = {}
    params["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    params["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw", "adagrad"])
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
    params["loss"] = trial.suggest_categorical("loss", ["bce", "focal", "weighted_bce"])
    if params["loss"] == "focal":
        params["gamma"] = trial.suggest_float("gamma", 0.5, 2.0)
    if params["loss"] == "weighted_bce":
        params["pos_weight"] = trial.suggest_float("pos_weight", 1.0, 3.0)  # capped at 3
    if name == "3-tower co-train":
        params["cotrain_weight"] = trial.suggest_float("cotrain_weight", 0.25, 1.0)
    return hp_from_params(default_hp, params, hp_keys)


class Objective:
    """What Optuna calls for every trial. A class, so it can remember the data and settings."""

    def __init__(self, name, default_hp, hp_keys, data, tune_fit, tune_val, device, seed):
        self.name = name
        self.default_hp = default_hp
        self.hp_keys = hp_keys
        self.data = data
        self.tune_fit = tune_fit
        self.tune_val = tune_val
        self.device = device
        self.seed = seed
        self.trial = None  # the trial that is running right now

    def report(self, check_number, auc):
        """Called after every validation check, so Optuna can stop a bad trial early."""
        self.trial.report(auc, check_number)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

    def __call__(self, trial):
        self.trial = trial
        hp = suggest(trial, self.name, self.default_hp, self.hp_keys)
        eval_rows = max(len(self.tune_fit) // 2, hp["batch_size"])  # 2 validation checks per trial
        model, best, curve = train_model(
            MODELS[self.name], hp, self.data, self.tune_fit, self.tune_val,
            1, eval_rows, 99, self.device, self.seed, self.report, verbose=False,
        )
        del model
        free_memory()
        return best["auc"]


def tune(name, default_hp, hp_keys, data, fit_rows, val_rows, device, seed, storage, n_trials, timeout_min,
         tune_rows, tune_val_rows):
    """Run (or continue) the Optuna search for one model. Returns the study."""
    study = optuna.create_study(
        study_name=slug(name), storage=storage, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )

    # How many trials already finished in an earlier run?
    finished = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.PRUNED:
            finished += 1
    if finished >= n_trials:
        return study

    # A brand new study starts with the default settings as trial 0
    if len(study.trials) == 0:
        first_trial = dict(DEFAULT_TRIAL)
        if name != "3-tower co-train":
            del first_trial["cotrain_weight"]
        study.enqueue_trial(first_trial)

    # Random subsets of the fit and validation rows, the same ones for every trial
    rng = np.random.default_rng(seed)
    tune_fit = np.sort(rng.choice(fit_rows, min(tune_rows, len(fit_rows)), replace=False))
    tune_val = np.sort(rng.choice(val_rows, min(tune_val_rows, len(val_rows)), replace=False))

    objective = Objective(name, default_hp, hp_keys, data, tune_fit, tune_val, device, seed)
    study.optimize(objective, n_trials=n_trials - finished, timeout=timeout_min * 60, catch=(RuntimeError,))
    return study


def should_adopt_tuned(study):
    """Use the tuned settings only if the best trial is not trial 0 (the default)."""
    return study.best_trial.number != 0
