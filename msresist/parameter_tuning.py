""" Hyperparameter Tuning using GridSearch. """

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from msresist.clustering import MassSpecClustering
from msresist.figures.figureM1 import IncorporateMissingValues, FitModelandComputeError


###------------ Building clustering method + PLSR pipeline and Tunning Hyperparameters ------------------###


def MSclusPLSR_tuning(X, info, Y, distance_method):
    """ Cross-validation: Simultaneous hyperparameter search. """
    MSclusPLSR = Pipeline(
        [
            ("MSclustering", MassSpecClustering(info=info, distance_method=distance_method, ncl=2, SeqWeight=0, n_runs=3, max_n_iter=200)),
            ("plsr", PLSRegression(n_components=2)),
        ]
    )
    param_grid = set_ClusterPLSRgrid()

    grid = GridSearchCV(MSclusPLSR, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring="neg_mean_squared_error")
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max


def set_ClusterPLSRgrid():
    """ Define the parameter combinations to test the clustering + plsr model with."""
    param_grid = []
    weights = [0.1, 1, 2, 3, 4, 5]
    for nn in range(2, 15):
        #         if nn < 5:
        param_grid.append(dict(MSclustering__ncl=[nn], MSclustering__SeqWeight=weights))
    #                                    plsr__n_components=list(np.arange(1, nn + 1))))
    #         if nn > 5:
    #             param_grid.append(dict(MSclustering__ncl=[nn],
    #                                    MSclustering__SeqWeight=weights,
    #                                    plsr__n_components=list(np.arange(1, 5))))
    return param_grid


def GridSearchCPTAC(x, distance_method, missingness):
    """Generate table with artifical missingness error across different hyperparameter combinations. Note
    that input is the portion of CPTAC data set without missing values."""
    assert True not in np.isnan(x.iloc[:, 4:]), "There are still NaNs."
    x.index = np.arange(x.shape[0])
    md, nan_indices = IncorporateMissingValues(x, missingness)
    ClusterToWeight = set_CPTACgrid()
    errors = []
    n_clusters = []
    weights_ = []
    SeqW, DatW, BothW, MixW = [], [], [], []
    for ncl, weights in ClusterToWeight.items():
        print("n_clusters:", ncl)
        for w in weights:
            print("weight:", w)
            error, wi = FitModelandComputeError(md, w, x, nan_indices, distance_method, ncl)
            weights_.append(w)
            n_clusters.append(ncl)
            errors.append(error)
            SeqW.append(wi[0])
            DatW.append(wi[1])
            BothW.append(wi[2])
            MixW.append(wi[3])

    X = pd.DataFrame()
    X["Weight"] = weights_
    X["n_clusters"] = n_clusters
    X["Error"] = errors
    X["SeqWins"] = SeqW
    X["DataWins"] = DatW
    X["BothWin"] = BothW
    X["MixWin"] = MixW
    return X


def set_CPTACgrid():
    """Define the parameter combinations to test the CPTAC model with."""
    ClusterToWeight = {}
    weights = [0, 0.075, 0.1, 0.15, 0.2, 1]
    for nn in range(5, 36, 3):
        ClusterToWeight[nn] = weights
    return ClusterToWeight