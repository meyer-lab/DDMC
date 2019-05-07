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
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, X, Y):
        self.cluster_assignments_ = KMeans(n_clusters=self.n_clusters).fit_predict(np.transpose(X))
#         self.kmeans = KMeans(n_clusters = self.n_clusters).fit(np.transpose(X))
        return self
        
    def transform(self,X):
        centers = ClusterAverages(X, self.cluster_assignments_, self.n_clusters, X.shape[0])
#         centers = np.array(self.kmeans.cluster_centers_).T
        return centers
    
class MyOwnRegressor(BaseEstimator):
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X, Y):   
        self.plsr = PLSRegression(n_components = self.n_components).fit(X,Y)
        return self
    
    def predict(self, X);
        # TODO: Should add assertions about the expected size of X, based upon training
        y_pred = self.plsr_.predict(X)
        return y_pred
    
    def score(self,X,Y):
        R2Y = self.plsr.score(X,Y)
        return R2Y 

    def Scores_Loadings(self, X, Y):
        X_scores, Y_scores = self.plsr.transform(X,Y)
        PC1_scores, PC2_scores = X_scores[:,0], X_scores[:,1]
        PC1_xload, PC2_xload = self.plsr.x_loadings_[:,0], plsr.x_loadings_[:,1]
        PC1_yload, PC2_yload = self.plsr.y_loadings_[:,0], plsr.y_loadings_[:,1]
        return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload
    
###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def TunningHyperpar(X,Y):
    estimators = [('kmeans', MyOwnKMEANS(2)), ('plsr', MyOwnRegressor(1))]
    pipe = Pipeline(estimators)
#     param_grid = dict(kmeans__n_clusters = np.arange(2,10), plsr__n_components = np.arange(2,10))
    param_grid = dict(kmeans__n_clusters = [2], plsr__n_components = [1,2]), dict(kmeans__n_clusters = [3], plsr__n_components = np.arange(1,4)), dict(kmeans__n_clusters = [4], plsr__n_components = np.arange(1,5)), dict(kmeans__n_clusters = [5], plsr__n_components = np.arange(1,6)), dict(kmeans__n_clusters = [6], plsr__n_components = np.arange(1,7)), dict(kmeans__n_clusters = [7], plsr__n_components = np.arange(1,8)), dict(kmeans__n_clusters = [8], plsr__n_components = np.arange(1,9)), dict(kmeans__n_clusters = [9], plsr__n_components = np.arange(1,10)), dict(kmeans__n_clusters = [10], plsr__n_components = np.arange(1,11)), dict(kmeans__n_clusters = [11], plsr__n_components = np.arange(1,12)), dict(kmeans__n_clusters = [12], plsr__n_components = np.arange(1,13)), dict(kmeans__n_clusters = [13], plsr__n_components = np.arange(1,14)), dict(kmeans__n_clusters = [14], plsr__n_components = np.arange(1,15)), dict(kmeans__n_clusters = [15], plsr__n_components = np.arange(1,16))
    grid = GridSearchCV(pipe, param_grid = param_grid, cv = X.shape[0], return_train_score = True)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = { '#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min, fit.best_params_


# param_grid = dict(a = [2], b = [1,2]), dict(a = [3], b = np.arange(1,4)), dict(a = [4], b = np.arange(1,5)), dict(a = [5], b = np.arange(1,6)), dict(a = [6], b = np.arange(1,7)), dict(a = [7], b = np.arange(1,8)), dict(a = [8], b = np.arange(1,9)), dict(a = [9], b = np.arange(1,10)), dict(a = [10], b = np.arange(1,11)), dict(a = [11], b = np.arange(1,12)), dict(a = [12], b = np.arange(1,13)), dict(a = [13], b = np.arange(1,14)), dict(a = [14], b = np.arange(1,15)), dict(a = [15], b = np.arange(1,16))