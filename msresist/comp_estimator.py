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

'''
- For some reason, only when using ClusterAverages() and not the built-in attribute .cluster_centers_, the returned centers are passed along to MyOnwRegressor
- The scoring method used in GridSearchCV for PLSR is giving erroneous high performance. Same is happening when using GridSearchCV with sklearn's PLSR alone (analysis_2estimators notebook).
- Tried to change the scoring method of GridSearch by specifying scoring = 'explained_variance', but same happened. 
'''

class MyOwnKMEANS(BaseEstimator):
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, X, Y):
        print('I ran!')
        self.cluster_assignments_ = KMeans(n_clusters=self.n_clusters).fit_predict(np.transpose(X))
        
    def transform(self, X):
        print('I ran too!')
        return ClusterAverages(X, self.cluster_assignments_, self.n_clusters, X.shape[0])
    
###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def ComHyperPar(X,Y):
    estimators = [('kmeans', MyOwnKMEANS(2)), ('plsr', PLSRegression(1))]
    pipe = Pipeline(estimators)
    param_grid = dict(kmeans__n_clusters = [2], plsr__n_components = [1,2]), dict(kmeans__n_clusters = [3], plsr__n_components = np.arange(1,4)), dict(kmeans__n_clusters = [4], plsr__n_components = np.arange(1,5)), dict(kmeans__n_clusters = [5], plsr__n_components = np.arange(1,6)), dict(kmeans__n_clusters = [6], plsr__n_components = np.arange(1,7)), dict(kmeans__n_clusters = [7], plsr__n_components = np.arange(1,8)), dict(kmeans__n_clusters = [8], plsr__n_components = np.arange(1,9)), dict(kmeans__n_clusters = [9], plsr__n_components = np.arange(1,10)), dict(kmeans__n_clusters = [10], plsr__n_components = np.arange(1,11)), dict(kmeans__n_clusters = [11], plsr__n_components = np.arange(1,12)), dict(kmeans__n_clusters = [12], plsr__n_components = np.arange(1,13)), dict(kmeans__n_clusters = [13], plsr__n_components = np.arange(1,14)), dict(kmeans__n_clusters = [14], plsr__n_components = np.arange(1,15)), dict(kmeans__n_clusters = [15], plsr__n_components = np.arange(1,16))
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=X.shape[0], return_train_score=True, verbose=100)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = { '#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min, fit.best_params_