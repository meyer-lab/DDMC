"Builds pipeline k-means + PLSR"

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from .plsr import ClusterAverages


class MyOwnKMEANS(BaseEstimator):
    """ Runs k-means providing the centers and cluster members and sequences """

    def __init__(self, n_clusters, ProtNames, peptide_phosphosite):
        """ define variables """
        self.n_clusters = n_clusters
        self.ProtNames = ProtNames
        self.peptide_phosphosite = peptide_phosphosite

    def fit(self, X, Y):
        """ fit data into k-means """
        self.kmeans_ = KMeans(n_clusters=self.n_clusters).fit(np.transpose(X))
        return self

    def transform(self, X):
        """ calculate cluster averages """
        centers, _ = ClusterAverages(np.array(X), self.kmeans_.labels_, self.n_clusters, X.shape[0], self.ProtNames, self.peptide_phosphosite)
        return centers

    def ClusterMembers(self, X):
        """ generate dictionary containing peptide names and sequences for each cluster """
        _, DictClusterToMembers = ClusterAverages(np.array(X), self.kmeans_.labels_, self.n_clusters, X.shape[0], self.ProtNames, self.peptide_phosphosite)
        return DictClusterToMembers


###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def ComHyperPar(X, Y, ProtNames, peptide_phosphosite):
    """ Cross-validation: Simultaneous hyperparameter search for number of clusters and number of components """
    estimators = [('kmeans', MyOwnKMEANS(5, ProtNames, peptide_phosphosite)), ('plsr', PLSRegression(2))]
    pipe = Pipeline(estimators)

    param_grid = []
    for nn in range(2, 16):
        param_grid.append(dict(kmeans__n_clusters=[nn], plsr__n_components=list(np.arange(1, nn + 1))))

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring='neg_mean_squared_error')
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {'#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'mean_test_scores': CVresults_max["mean_test_score"], 'mean_train_scores': CVresults_max["mean_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min, fit.best_params_
