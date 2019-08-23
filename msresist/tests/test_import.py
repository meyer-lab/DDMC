"""
Testing file for the chained methods.
"""
import unittest
import numpy as np
import pandas as pd
from ..pre_processing import preprocessing

class TestImport(unittest.TestCase):
    """ Testing class data import functions. """

    def test_import(self):
        """ Test that we get reasonable values from data import. """

        # If we don't have log2-transformed values, none should be negative
        ABC_mc1 = preprocessing(motifs=True, FCfilter=True, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc1.iloc[:, 2:])) > 0.0)

        ABC_mc2 = preprocessing(motifs=True, FCfilter=False, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc2.iloc[:, 2:])) > 0.0)

        ABC_mc3 = preprocessing(motifs=False, FCfilter=True, log2T=False)
        self.assertTrue(np.all(np.min(ABC_mc3.iloc[:, 2:])) > 0.0)

        # Length should go down with filtering
        self.assertGreater(len(ABC_mc2.index), len(ABC_mc1.index))
