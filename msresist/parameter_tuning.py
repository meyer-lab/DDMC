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
    MSclusPLSR = Pipeline([("MSclustering", MassSpecClustering(info=info, 
                                                               distance_method=distance_method, 
                                                               ncl=2, 
                                                               SeqWeight=0,
                                                               n_runs=3)),
                           ("plsr", PLSRegression(n_components=2))])
    param_grid = set_grid()

    grid = GridSearchCV(MSclusPLSR, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring="neg_mean_squared_error")
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {
        "#Clusters": CVresults_max["param_MSclustering__ncl"],
#         "#Components": CVresults_max["param_plsr__n_components"],
        "SeqWeights": CVresults_max["param_MSclustering__SeqWeight"],
        "mean_test_scores": CVresults_max["mean_test_score"],
        "mean_train_scores": CVresults_max["mean_train_score"],
    }
    return CVresults_max

###------------ General GridSearch Structure ------------------###


def GridSearch_CV(model, parameters, cv, X, Y=None, scoring=None):
    """ Exhaustive search over specified parameter values for an estimator. """
    grid = GridSearchCV(model, param_grid=parameters, cv=cv, scoring=scoring)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max


def set_grid():
    """ Define the parameter combinations to test the model with. """
    param_grid = []
    weights = [0.5, 15, 50, 100]
    for nn in range(2, 11):
#         if nn < 5:
        param_grid.append(dict(MSclustering__ncl=[nn],
                               MSclustering__SeqWeight=weights))
#                                    plsr__n_components=list(np.arange(1, nn + 1))))
#         if nn > 5:
#             param_grid.append(dict(MSclustering__ncl=[nn],
#                                    MSclustering__SeqWeight=weights,
#                                    plsr__n_components=list(np.arange(1, 5))))
    return param_grid
