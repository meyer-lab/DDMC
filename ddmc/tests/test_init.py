import numpy as np
import pytest
import sklearn.utils


def test_check_array_translates_force_all_finite_kwarg():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    # The patched check_array (installed at package-import time) must accept
    # the old `force_all_finite` kwarg that fancyimpute still passes.
    out = sklearn.utils.check_array(X, force_all_finite=True)

    np.testing.assert_array_equal(out, X)


def test_check_array_still_rejects_non_finite_values():
    X = np.array([[1.0, np.nan]])

    with pytest.raises(ValueError):
        sklearn.utils.check_array(X, force_all_finite=True)
