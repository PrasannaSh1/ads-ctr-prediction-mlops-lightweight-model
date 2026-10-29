"""Time-ordered click-rate encodings (notebook section 4.4, constraint 3 in
PROMPT_taobao_ctr_mlops.md): a fit row's encoding must never depend on its own label, and
validation must be encoded exactly like the test day — frozen at the end of the fit period,
while test additionally sees the validation period."""
from __future__ import annotations

import numpy as np

from taobao_ctr.features.click_rates import click_rate_features

TE_USER = 0  # first column of click_rate_features' output, per TE_COLS


def _toy_imp_and_X():
    # time order == array index. User 1 at fit rows 0,1; user 2 at fit rows 2,3;
    # user 3 appears only at row 4 (val) and row 5 (test) -- unseen during fit.
    u = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
    y = np.array([1, 0, 0, 1, 1, 0], dtype=np.float32)
    zeros = np.zeros(len(u), dtype=np.int64)
    X = {"adgroup_id": zeros, "campaign_id": zeros, "customer": zeros, "ad_brand": zeros, "ad_cate": zeros}
    imp = {"u": u, "y": y, "price": np.ones(len(u), dtype=np.float32)}
    fit_rows, val_rows, test_rows = np.array([0, 1, 2, 3]), np.array([4]), np.array([5])
    return imp, X, fit_rows, val_rows, test_rows


def test_fit_rows_never_use_their_own_label():
    imp, X, fit_rows, val_rows, test_rows = _toy_imp_and_X()
    te = click_rate_features(imp, X, fit_rows, val_rows, test_rows, n_cate=1, te_alpha=20)

    # Swap the LAST fit row's label of each user (row 1 for user 1, row 3 for user 2). Each
    # row's own encoding only ever uses *earlier* same-key rows, so this must leave every fit
    # row's encoding completely unchanged -- if it changed, the row would be leaking its own label.
    y_swapped = imp["y"].copy()
    y_swapped[1], y_swapped[3] = y_swapped[3], y_swapped[1]
    te_swapped = click_rate_features(
        {**imp, "y": y_swapped}, X, fit_rows, val_rows, test_rows, n_cate=1, te_alpha=20,
    )
    np.testing.assert_array_equal(
        te[fit_rows, TE_USER], te_swapped[fit_rows, TE_USER],
        err_msg="a fit row's click-rate encoding must not depend on its own label",
    )


def test_unseen_key_falls_back_to_prior_in_fit_and_val_but_not_in_test():
    imp, X, fit_rows, val_rows, test_rows = _toy_imp_and_X()
    te = click_rate_features(imp, X, fit_rows, val_rows, test_rows, n_cate=1, te_alpha=20)

    # Row 0 and row 2 are each user's very first fit-time appearance: zero history -> exactly
    # the global prior, regardless of their own (different) labels.
    assert te[0, TE_USER] == te[2, TE_USER]

    # User 3 never appears in fit_rows, so the validation row (row 4) must fall back to that
    # same prior -- it is encoded exactly like a brand-new fit-time user, never touching its own label.
    assert te[4, TE_USER] == te[0, TE_USER]

    # The test row (row 5, same user 3) is encoded from fit+val ("all training days"), which by
    # now includes row 4 -- so it must differ from the frozen validation-time value.
    assert te[5, TE_USER] != te[4, TE_USER]
