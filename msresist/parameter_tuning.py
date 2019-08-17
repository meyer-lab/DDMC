""" Hyperparameter Tuning using GridSearch. """

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from msresist.clustering import MyOwnKMEANS

###------------ Building k-means + PLSR pipeline and Tunning Hyperparameters ------------------###

def ComHyperPar(X, Y):
    """ Cross-validation: Simultaneous hyperparameter search for number of clusters and number of components """
    estimators = [('kmeans', MyOwnKMEANS(5)), ('plsr', PLSRegression(2))]
    pipe = Pipeline(estimators)

    param_grid = []
    for nn in range(2, 16):
        param_grid.append(dict(kmeans__n_clusters=[nn], plsr__n_components=list(np.arange(1, nn + 1))))

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring='neg_mean_squared_error')
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {'#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'mean_test_scores': CVresults_max["mean_test_score"], 'mean_train_scores': CVresults_max["mean_train_score"]}
    return CVresults_max, pd.DataFrame(data=std_scores), fit.best_params_



###------------ General GridSearch Structure ------------------###

def GridSearch_CV(model, X, Y, parameters, cv, scoring=None):
    """ Exhaustive search over specified parameter values for an estimator. """
    grid = GridSearchCV(model, param_grid=parameters, cv=cv, scoring=scoring)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max
