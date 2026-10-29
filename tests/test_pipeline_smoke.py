"""Whole pipeline end to end on synthetic data, tiny settings: build the dataset from the raw
CSVs, train one deep model and LightGBM, and confirm both land well above chance (AUC 0.5) --
the real synthetic-data generator is tuned to make a *correctly wired* pipeline reach roughly
0.63-0.65, so a passing run here is evidence the plumbing (encodings, history, batching, loss)
is connected correctly end to end, not just that each piece works in isolation.
"""
from __future__ import annotations

from taobao_ctr.data.dataset import build_dataset
from taobao_ctr.models.registry import MODELS
from taobao_ctr.training.lightgbm_train import lgb_predict, train_lightgbm
from taobao_ctr.training.trainer import default_hp, get_device, metrics, predict, train_model

MIN_AUC = 0.55


def _build(tmp_path, params):
    return build_dataset(
        data_dir=tmp_path, kaggle_dataset="unused/unused",
        n_buckets=params.data.n_buckets, chunk_rows=params.data.chunk_rows,
        seq_len=params.data.seq_len, te_alpha=params.features.te_alpha,
    )


def test_deep_model_reaches_reasonable_auc(tmp_path, patch_kaggle_csv_smoke, synthetic_params):
    params = synthetic_params
    dataset, _train_rows, fit_rows, val_rows, test_rows = _build(tmp_path, params)
    device = get_device()
    hp = default_hp(params)
    hp["batch_size"] = 2048  # bigger batches -> fewer Python-level steps per epoch on CPU

    model, _best, _curve = train_model(
        MODELS["Two-tower"], hp, dataset, fit_rows, val_rows,
        # one epoch, one validation check at the end (eval_every_rows set past the fit set size)
        max_epochs=1, eval_every_rows=len(fit_rows) + 1, patience=hp["patience"],
        device=device, seed=params.seed, verbose=False,
    )
    p_test = predict(model, dataset, test_rows, device)
    m = metrics(dataset.y[test_rows], p_test["logit"])
    assert m["AUC"] > MIN_AUC, m


def test_lightgbm_reaches_reasonable_auc(tmp_path, patch_kaggle_csv_smoke, synthetic_params):
    params = synthetic_params
    dataset, _train_rows, fit_rows, val_rows, test_rows = _build(tmp_path, params)
    lgb_cfg = params.train.lightgbm
    lgb_params = {
        "objective": "binary", "learning_rate": lgb_cfg.learning_rate, "num_leaves": lgb_cfg.num_leaves,
        "min_data_in_leaf": lgb_cfg.min_data_in_leaf, "feature_fraction": lgb_cfg.feature_fraction,
        "bagging_fraction": lgb_cfg.bagging_fraction, "bagging_freq": lgb_cfg.bagging_freq,
        "metric": "auc", "verbose": -1, "seed": params.seed,
    }
    gbm = train_lightgbm(
        dataset, fit_rows, val_rows, lgb_params,
        num_boost_round=lgb_cfg.num_boost_round, early_stopping_rounds=lgb_cfg.early_stopping_rounds,
        log_evaluation_every=0,
    )
    p_test = lgb_predict(gbm, dataset, test_rows)
    m = metrics(dataset.y[test_rows], p_test)
    assert m["AUC"] > MIN_AUC, m
