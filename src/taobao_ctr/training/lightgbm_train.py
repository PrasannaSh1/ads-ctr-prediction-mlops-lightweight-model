"""FeatureSequence, LightGBM training + predict (notebook section 8). Rows are fed to LightGBM
through `lgb.Sequence` in batches, so the full feature matrix never sits in RAM at once."""
from __future__ import annotations

import numbers

import lightgbm as lgb
import numpy as np

from taobao_ctr.data.dataset import Dataset
from taobao_ctr.features.interactions import LGB_FEATURES, LOWCARD, lgb_features


class FeatureSequence(lgb.Sequence):
    batch_size = 500_000

    def __init__(self, data: Dataset, rows: np.ndarray):
        self.data, self.rows = data, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            return lgb_features(self.data, self.rows[[i]])[0]
        return lgb_features(self.data, self.rows[i])


def lgb_predict(model: lgb.Booster, data: Dataset, rows: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    return np.concatenate([
        model.predict(lgb_features(data, rows[s:s + chunk]), num_iteration=model.best_iteration)
        for s in range(0, len(rows), chunk)
    ])


def train_lightgbm(
    data: Dataset, fit_rows: np.ndarray, val_rows: np.ndarray, params: dict,
    num_boost_round: int = 1000, early_stopping_rounds: int = 50, log_evaluation_every: int = 50,
) -> lgb.Booster:
    d_fit = lgb.Dataset(
        FeatureSequence(data, fit_rows), data.y[fit_rows], feature_name=LGB_FEATURES,
        categorical_feature=LOWCARD, free_raw_data=True,
    )
    d_val = lgb.Dataset(FeatureSequence(data, val_rows), data.y[val_rows], reference=d_fit)
    return lgb.train(
        params, d_fit, num_boost_round=num_boost_round, valid_sets=[d_val],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(log_evaluation_every)],
    )
