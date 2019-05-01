import numpy as np, pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin, ClusterMixin, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from PLSR_functions import FilteringOutPeptides, ClusterAverages
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV


###------------ Creating Own Estimators ------------------###
'''
Unresolved issues / questions: 
    - Q2Y method, currently only R2Y
    - Fit only X_train and y_train? 
    - Error in Scores_Loadings: operands could not be broadcast together with shapes (10,5) (96,) (10,5). self.plsr_ not correcly imported?
'''

class MyEstimator(BaseEstimator):
    
    def __init__(self, n_clusters, n_components):
        self.n_clusters = n_clusters
        self.n_components = n_components
    
    def fit(self, X, Y):
#         self.cluster_assignments_ = KMeans(n_clusters=self.n_clusters).fit_predict(np.transpose(X))
        self.kmeans_ = KMeans(n_clusters=self.n_clusters).fit(np.transpose(X))
        self.plsr_ = PLSRegression(n_components = self.n_components).fit(X,Y)
        self.X_ = X
        self.Y_ = Y
        return self
        
    def transform(self,X):
        check_is_fitted(self, ['X_', 'Y_'])
        X = check_array(X)
        self.cluster_assignments_ = self.kmeans_.predict(np.transpose(X))
        X_Filt_Clust_Avgs = ClusterAverages(X, self.cluster_assignments_, self.n_clusters, 10)
        return X_Filt_Clust_Avgs
    
    def score(self,X,Y):
        #KMscore = self.kmeans_.score(self.X_)
        R2Y = self.plsr_.score(X,Y)
        return R2Y #,KMscore
        
    def Scores_Loadings(self, X, Y):
        X_scores, Y_scores = self.plsr_.transform(X, Y)
        PC1_scores, PC2_scores = X_scores[:,0], X_scores[:,1]
        PC1_xload, PC2_xload = plsr.x_loadings_[:,0], plsr.x_loadings_[:,1]
        PC1_yload, PC2_yload = plsr.y_loadings_[:,0], plsr.y_loadings_[:,1]
        return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload
    
###-------------- Tunning Hyperparameters ------------------###

def TunningHyperpar(X,Y):
    parameters = {'n_clusters': np.arange(2,11), 'n_components': np.arange(2,11)}
    grid = GridSearchCV(MyEstimator(2,2), parameters, cv=X.shape[0])
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = { '#Clusters': CVresults_max['param_n_clusters'], '#Components': CVresults_max['param_n_components'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min
