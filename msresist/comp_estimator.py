import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, GridSearchCV, ParameterGrid
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.utils import check_consistent_length
from .plsr import ClusterAverages
from sklearn.pipeline import Pipeline


class MyOwnKMEANS(BaseEstimator):
    
    def __init__(self, n_clusters, ProtNames, peptide_phosphosite):
        self.n_clusters = n_clusters
        self.ProtNames = ProtNames
        self.peptide_phosphosite = peptide_phosphosite
    
    def fit(self, X, Y):
#         X_f = 
        self.kmeans_ = KMeans(n_clusters=self.n_clusters).fit(np.transpose(X))
        return self

    def transform(self, X):
        centers, DictClusterToMembers = ClusterAverages(np.array(X), self.kmeans_.labels_, self.n_clusters, X.shape[0], self.ProtNames, self.peptide_phosphosite)
        return centers 
    
    def ClusterMembers(self, X):
        centers, DictClusterToMembers = ClusterAverages(np.array(X), self.kmeans_.labels_, self.n_clusters, X.shape[0], self.ProtNames, self.peptide_phosphosite)
        return DictClusterToMembers
                                                              
                                                              
###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def ComHyperPar(X,Y, ProtNames, peptide_phosphosite):
    estimators = [('kmeans', MyOwnKMEANS(5, ProtNames, peptide_phosphosite)), ('plsr', PLSRegression(2))]
    pipe = Pipeline(estimators)
    param_grid = dict(kmeans__n_clusters = [2], plsr__n_components = [1,2]), dict(kmeans__n_clusters = [3], plsr__n_components = np.arange(1,4)), dict(kmeans__n_clusters = [4], plsr__n_components = np.arange(1,5)), dict(kmeans__n_clusters = [5], plsr__n_components = np.arange(1,6)), dict(kmeans__n_clusters = [6], plsr__n_components = np.arange(1,7)), dict(kmeans__n_clusters = [7], plsr__n_components = np.arange(1,8)), dict(kmeans__n_clusters = [8], plsr__n_components = np.arange(1,9)), dict(kmeans__n_clusters = [9], plsr__n_components = np.arange(1,10)), dict(kmeans__n_clusters = [10], plsr__n_components = np.arange(1,11)), dict(kmeans__n_clusters = [11], plsr__n_components = np.arange(1,12)), dict(kmeans__n_clusters = [12], plsr__n_components = np.arange(1,13)), dict(kmeans__n_clusters = [13], plsr__n_components = np.arange(1,14)), dict(kmeans__n_clusters = [14], plsr__n_components = np.arange(1,15)), dict(kmeans__n_clusters = [15], plsr__n_components = np.arange(1,16))
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=X.shape[0], return_train_score=True, scoring='neg_mean_squared_error')
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = { '#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'mean_test_scores': CVresults_max["mean_test_score"], 'mean_train_scores': CVresults_max["mean_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min, fit.best_params_

                                                              