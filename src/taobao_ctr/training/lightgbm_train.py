"""LightGBM training and prediction.

The rows are given to LightGBM through a Sequence, so the feature matrix is built in pieces
and never sits in RAM all at once.
"""
import numbers

import lightgbm as lgb
import numpy as np

from taobao_ctr.features.interactions import LGB_FEATURES, LOWCARD, lgb_features


class FeatureSequence(lgb.Sequence):
    """Lets LightGBM ask for the features of some rows whenever it needs them."""

    batch_size = 500000

    def __init__(self, data, rows):
        self.data = data
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):  # a single row number (Python or numpy integer)
            return lgb_features(self.data, self.rows[[i]])[0]
        return lgb_features(self.data, self.rows[i])  # a slice of row numbers


def lgb_predict(model, data, rows, chunk=500000):
    """Click probabilities for the given rows, computed in chunks."""
    pieces = []
    for start in range(0, len(rows), chunk):
        features = lgb_features(data, rows[start:start + chunk])
        pieces.append(model.predict(features, num_iteration=model.best_iteration))
    return np.concatenate(pieces)


def train_lightgbm(data, fit_rows, val_rows, params, num_boost_round=1000, early_stopping_rounds=50,
                   log_evaluation_every=50):
    d_fit = lgb.Dataset(
        FeatureSequence(data, fit_rows), data.y[fit_rows], feature_name=LGB_FEATURES,
        categorical_feature=LOWCARD, free_raw_data=True,
    )
    d_val = lgb.Dataset(FeatureSequence(data, val_rows), data.y[val_rows], reference=d_fit)
    callbacks = [lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(log_evaluation_every)]
    model = lgb.train(params, d_fit, num_boost_round=num_boost_round, valid_sets=[d_val], callbacks=callbacks)
    return model
