"""This is the __init__.py file."""

import sklearn.utils

__version__ = "0.0.1"

# fancyimpute (unmaintained since 2020) calls check_array with the
# `force_all_finite` kwarg, which scikit-learn renamed to `ensure_all_finite`
# and later removed. Patch it here, at package-import time, so it is in place
# before any submodule (however it orders its own imports) pulls in fancyimpute.
_sklearn_check_array = sklearn.utils.check_array


def _check_array_compat(X, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _sklearn_check_array(X, **kwargs)


sklearn.utils.check_array = _check_array_compat  # ty: ignore[invalid-assignment]
