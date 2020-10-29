"""
This creates Figure M3.
"""

import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from ..figures.figureM2 import TumorType
from ..logistic_regression import plotClusterCoefficients, plotPredictionProbabilities, plotConfusionMatrix, plotROC
from ..figures.figure3 import plotPCA
from ..clustering import MassSpecClustering
from msresist.pre_processing import filter_NaNpeptides
import pickle


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 3))

    

    return f
