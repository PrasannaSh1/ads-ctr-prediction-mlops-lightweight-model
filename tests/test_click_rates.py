"""Checks the click-rate features: a fit row must never use its own label, validation uses
the rates known at the end of the fit period, and test also sees the validation period."""
import numpy as np

from taobao_ctr.features.click_rates import click_rate_features

TE_USER = 0  # the first column is the user's click rate (see TE_COLS)


def toy_data():
    # Time order = array index. User 1 is in fit rows 0 and 1; user 2 in fit rows 2 and 3;
    # user 3 only appears in row 4 (validation) and row 5 (test), so it is unseen during fit.
    u = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
    y = np.array([1, 0, 0, 1, 1, 0], dtype=np.float32)
    zeros = np.zeros(len(u), dtype=np.int64)
    X = {"adgroup_id": zeros, "campaign_id": zeros, "customer": zeros, "ad_brand": zeros, "ad_cate": zeros}
    imp = {"u": u, "y": y, "price": np.ones(len(u), dtype=np.float32)}
    fit_rows = np.array([0, 1, 2, 3])
    val_rows = np.array([4])
    test_rows = np.array([5])
    return imp, X, fit_rows, val_rows, test_rows


def test_fit_rows_never_use_their_own_label():
    imp, X, fit_rows, val_rows, test_rows = toy_data()
    te = click_rate_features(imp, X, fit_rows, val_rows, test_rows, n_cate=1, te_alpha=20)

    # Swap the LAST fit row's label of each user (row 1 for user 1, row 3 for user 2).
    # A row's encoding only uses EARLIER rows of the same user, so nothing may change.
    y_swapped = imp["y"].copy()
    y_swapped[1], y_swapped[3] = y_swapped[3], y_swapped[1]
    imp_swapped = dict(imp)
    imp_swapped["y"] = y_swapped
    te_swapped = click_rate_features(imp_swapped, X, fit_rows, val_rows, test_rows, n_cate=1, te_alpha=20)

    np.testing.assert_array_equal(
        te[fit_rows, TE_USER], te_swapped[fit_rows, TE_USER],
        err_msg="a fit row's click-rate encoding must not depend on its own label",
    )


def test_unseen_key_falls_back_to_prior_in_fit_and_val_but_not_in_test():
    imp, X, fit_rows, val_rows, test_rows = toy_data()
    te = click_rate_features(imp, X, fit_rows, val_rows, test_rows, n_cate=1, te_alpha=20)

    # Row 0 and row 2 are each user's first appearance: no history, so both get exactly the prior
    assert te[0, TE_USER] == te[2, TE_USER]

    # User 3 is never in the fit rows, so its validation row (row 4) also gets the prior
    assert te[4, TE_USER] == te[0, TE_USER]

    # The test row (row 5, same user 3) is encoded from fit + validation, which now includes row 4,
    # so it must differ from the validation value
    assert te[5, TE_USER] != te[4, TE_USER]
