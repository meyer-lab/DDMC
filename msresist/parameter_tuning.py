""" Hyperparameter Tuning using GridSearch. """

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from .clustering import MyOwnKMEANS, MyOwnGMM, MassSpecClustering


###------------ Building clustering method + PLSR pipeline and Tunning Hyperparameters ------------------###


def kmeansPLSR_tuning(X, Y):
    """ Cross-validation: Simultaneous hyperparameter search for number of clusters for k-means and number of components for PLSR """
    kmeansPLSR = Pipeline([("kmeans", MyOwnKMEANS(5)), ("plsr", PLSRegression(2))])

    param_grid = []
    for nn in range(2, 16):
        param_grid.append(dict(kmeans__n_clusters=[nn], plsr__n_components=list(np.arange(1, nn + 1))))

    grid = GridSearchCV(kmeansPLSR, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring="neg_mean_squared_error")
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


def MSclusPLSR_tuning(X, seqs, names, Y):
    MSclusPLSR = Pipeline([("MSclustering", MassSpecClustering(seqs=seqs, names=names, ncl=5)), ("plsr", PLSRegression(n_components=2))])

    param_grid = []
    for nn in range(2, 8):
        param_grid.append(dict(MSclustering__ncl=[nn], MSclustering__GMMweight=[0.5, 2.5, 5],
                               plsr__n_components=list(np.arange(1, nn + 1))))

    grid = GridSearchCV(MSclusPLSR, param_grid=param_grid, cv=10, return_train_score=True, scoring="neg_mean_squared_error")
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {
        "Ranking": CVresults_max["rank_test_score"],
        "#Clusters": CVresults_max["param_MSclustering__ncl"],
        "GMMweights": CVresults_max["param_MSclustering__GMMweight"],
        "#ComponentsPLSR": CVresults_max["param_plsr__n_components"],
        "mean_test_scores": CVresults_max["mean_test_score"],
        "mean_train_scores": CVresults_max["mean_train_score"],
    }
    return CVresults_max, pd.DataFrame(data=std_scores), fit.best_params_


###------------ General GridSearch Structure ------------------###


def GridSearch_CV(model, parameters, cv, X, Y=None, scoring=None):
    """ Exhaustive search over specified parameter values for an estimator. """
    grid = GridSearchCV(model, param_grid=parameters, cv=cv, scoring=scoring)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max
