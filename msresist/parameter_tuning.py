""" Hyperparameter Tuning using GridSearch. """

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from msresist.clustering import MassSpecClustering


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
    weights = [0.1, 1, 3, 5, 7, 50]
    for nn in range(2, 15):
        #         if nn < 5:
        param_grid.append(dict(MSclustering__ncl=[nn], MSclustering__SeqWeight=weights,
                                       plsr__n_components=list(np.arange(1, nn + 1))))
        if nn > 5:
            param_grid.append(dict(MSclustering__ncl=[nn],
                                   MSclustering__SeqWeight=weights,
                                   plsr__n_components=list(np.arange(1, 5))))
    return param_grid
