"""
Testing file for the chained methods.
"""
import numpy as np
from ..pre_processing import preprocessing


def test_import():
    """ Test that we get reasonable values from data import. """
    # If we don't have log2-transformed values, none should be negative
    ABC_mc1 = preprocessing(AXLwt_GF=True, FCfilter=True, log2T=False, mc_row=True)
    assert np.all(ABC_mc1.min(numeric_only=True) >= 0.0)

    ABC_mc2 = preprocessing(AXLwt_GF=True, FCfilter=False, log2T=False, mc_row=True)
    assert np.all(ABC_mc2.min(numeric_only=True) >= 0.0)

    # If we don't have log2-transformed values, none should be negative
    ABC_mc3 = preprocessing(AXLwt_GF=True, Vfilter=True, FCfilter=True, log2T=False, mc_row=True)
    assert np.all(ABC_mc3.min(numeric_only=True) >= 0.0)

    ABC_mc4 = preprocessing(AXLwt_GF=True, Vfilter=True, FCfilter=False, log2T=False, mc_row=True)
    assert np.all(ABC_mc4.min(numeric_only=True) >= 0.0)

    # Length should go down with filtering
    assert len(ABC_mc2.index) > len(ABC_mc1.index)
