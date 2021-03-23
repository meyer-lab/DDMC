""" Hyperparameter Tuning using GridSearch. """

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from msresist.clustering import DDMC


###------------ Building clustering method + PLSR pipeline and Tunning Hyperparameters ------------------###


def DDMC_PLSR_tuning(X, info, Y, distance_method, weights, clusters):
    """ Cross-validation: Simultaneous hyperparameter search. """
    MSclusPLSR = Pipeline(
        [
            ("MSclustering", DDMC(info=info, distance_method=distance_method, ncl=2, SeqWeight=0)),
            ("plsr", PLSRegression(n_components=2)),
        ]
    )
    param_grid = set_grid(weights, clusters)

    grid = GridSearchCV(MSclusPLSR, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring="neg_mean_squared_error")
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max


def set_grid(weights, clusters):
    """ Define the parameter combinations to test the clustering + plsr model with."""
    param_grid = []
    for nn in range(2, clusters):
        if nn <= 5:
            param_grid.append(dict(MSclustering__ncl=[nn], MSclustering__SeqWeight=weights,
                               plsr__n_components=list(np.arange(1, nn + 1))))
        elif nn > 5:
            param_grid.append(dict(MSclustering__ncl=[nn],
                                   MSclustering__SeqWeight=weights,
                                   plsr__n_components=list(np.arange(1, 5))))
    return param_grid
