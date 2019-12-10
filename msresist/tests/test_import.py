"""
Testing file for the chained methods.
"""
import unittest
import numpy as np
from ..pre_processing import preprocessing


class TestImport(unittest.TestCase):
    """ Testing class data import functions. """

    def test_import(self):
        """ Test that we get reasonable values from data import. """

        # If we don't have log2-transformed values, none should be negative
        ABC_mc1 = preprocessing(AXLwt=True, motifs=True, FCfilter=True, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc1.iloc[:, 5:])) > 0.0)

        ABC_mc2 = preprocessing(AXLwt=True, motifs=True, FCfilter=False, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc2.iloc[:, 5:])) > 0.0)

        ABC_mc3 = preprocessing(AXLwt=True, motifs=False, FCfilter=True, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc3.iloc[:, 5:])) > 0.0)

        # If we don't have log2-transformed values, none should be negative
        ABC_mc4 = preprocessing(AXLwt=True, Vfilter=True, motifs=True, FCfilter=True, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc4.iloc[:, 5:])) > 0.0)

        ABC_mc5 = preprocessing(AXLwt=True, Vfilter=True, motifs=True, FCfilter=False, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc5.iloc[:, 5:])) > 0.0)

        ABC_mc6 = preprocessing(AXLwt=True, Vfilter=True, motifs=False, FCfilter=True, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc6.iloc[:, 5:])) > 0.0)

        # Length should go down with filtering
        self.assertGreater(len(ABC_mc2.index), len(ABC_mc1.index))
