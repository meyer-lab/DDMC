import pytest

from ddmc.datasets import CPTAC, filter_incomplete_peptides


@pytest.fixture(scope="session")
def p_signal():
    """Loading and filtering CPTAC's p-signal data is expensive and the
    result is never mutated by DDMC.fit, so share one copy across the whole
    test session instead of reloading it in every test."""
    return filter_incomplete_peptides(CPTAC().get_p_signal(), sample_presence_ratio=1)
