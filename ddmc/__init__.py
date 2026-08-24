"""Package entry point for `ddmc`.

Sets `__version__`, and patches scikit-learn's `check_array` so the
unmaintained `fancyimpute` dependency (used by `ddmc.clustering.DDMC` for
missing-value imputation) keeps working against modern scikit-learn.
"""

import sklearn.utils

__version__ = "0.0.1"

# fancyimpute (unmaintained since 2020) calls check_array with the
# `force_all_finite` kwarg, which scikit-learn renamed to `ensure_all_finite`
# and later removed. Patch it here, at package-import time, so it is in place
# before any submodule (however it orders its own imports) pulls in fancyimpute.
_sklearn_check_array = sklearn.utils.check_array


def _check_array_compat(X, **kwargs):
    """Translate the removed `force_all_finite` kwarg to `ensure_all_finite` and
    delegate to the original `sklearn.utils.check_array`.

    Args:
        X: The array-like to validate; forwarded unchanged.
        **kwargs: Keyword arguments for `check_array`. If `force_all_finite`
            is present, it is renamed to `ensure_all_finite`.

    Returns:
        The validated array, as returned by the original `check_array`.
    """
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _sklearn_check_array(X, **kwargs)


sklearn.utils.check_array = _check_array_compat  # ty: ignore[invalid-assignment]
