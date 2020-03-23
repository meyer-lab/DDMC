""" Hyperparameter Tuning using GridSearch. """

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from .clustering import MyOwnKMEANS, MassSpecClustering


###------------ Building clustering method + PLSR pipeline and Tunning Hyperparameters ------------------###


def kmeansPLSR_tuning(X, Y):
    """ Cross-validation: Simultaneous hyperparameter search for number of clusters for k-means and number of components for PLSR. """
    kmeansPLSR = Pipeline([("kmeans", MyOwnKMEANS(5)), ("plsr", PLSRegression(2))])

    param_grid = []
    for nn in range(2, 16):
        param_grid.append(dict(kmeans__n_clusters=[nn], plsr__n_components=list(np.arange(1, nn + 1))))

    grid = GridSearchCV(kmeansPLSR, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring="neg_mean_squared_error", n_jobs=2)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {
        "Ranking": CVresults_max["rank_test_score"],
        "#Clusters": CVresults_max["param_kmeans__n_clusters"],
        "#Components": CVresults_max["param_plsr__n_components"],
        "mean_test_scores": CVresults_max["mean_test_score"],
        "mean_train_scores": CVresults_max["mean_train_score"],
    }
    return CVresults_max, pd.DataFrame(data=std_scores), fit.best_params_


def MSclusPLSR_tuning(X, info, Y, distance_method):
    """ Cross-validation: Simultaneous hyperparameter search. """
    MSclusPLSR = Pipeline([("MSclustering", MassSpecClustering(info=info, distance_method=distance_method, ncl=2, GMMweight=0.5)), ("plsr", PLSRegression(n_components=2))])
    param_grid = set_grid()

    grid = GridSearchCV(MSclusPLSR, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring="neg_mean_squared_error", n_jobs=-1)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {
        "#Clusters": CVresults_max["param_MSclustering__ncl"],
        "#Components": CVresults_max["param_plsr__n_components"],
        "GMMweights": CVresults_max["param_MSclustering__GMMweight"],
        "mean_test_scores": CVresults_max["mean_test_score"],
        "mean_train_scores": CVresults_max["mean_train_score"],
    }
    return std_scores

###------------ General GridSearch Structure ------------------###


def GridSearch_CV(model, parameters, cv, X, Y=None, scoring=None):
    """ Exhaustive search over specified parameter values for an estimator. """
    grid = GridSearchCV(model, param_grid=parameters, cv=cv, scoring=scoring, n_jobs=-1)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max


def set_grid():
    param_grid = []
    for nn in range(2, 16):
        if nn < 5:
            param_grid.append(dict(MSclustering__ncl=[nn],
                                   MSclustering__GMMweight=[0.0, 0.1, 0.25, 0.5, 1, 5, 10, 20],
                                   plsr__n_components=list(np.arange(1, nn + 1))))
        if nn > 5:
            param_grid.append(dict(MSclustering__ncl=[nn],
                                   MSclustering__GMMweight=[0.0, 0.1, 0.25, 0.5, 1, 5, 10, 20],
                                   plsr__n_components=list(np.arange(1, 5))))
    return param_grid
