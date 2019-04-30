import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin, ClusterMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from PLSR_functions import FilteringOutPeptides, ClusterAverages
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV


###------------ Creating Own Estimators ------------------###
'''
Unresolved issues / questions: 
    - Structure: Should both estimators be in same def fit? Flow of analysis broken
    - RegressorMixin / ClusterMixin necessary unless I specify name of functions "transform", "score"
    - First row missing in fit = grid.fit(X,Y). CV?
    - CV runs across conditions instead of peptides shape = 10:nClusters
    - Current error... Incorrect number of features. Got 1 features, expected 9
'''

class MyOwnKMEANS(BaseEstimator):
    
    def __init__(self, n_clusters=2, n_components = 2):
        self.n_clusters = n_clusters
        self.n_components = n_components
    
    def fit(self, X, Y):
        print("fit", X)
        self.kmeans = KMeans(n_clusters=self.n_clusters).fit(np.transpose(X))
        return self
        
    def transform(self, X, Y):
        print(X)
        raise SystemExit
        cluster_assignments = self.kmeans.predict(np.transpose(X))
        self.X_Filt_Clust_Avgs = ClusterAverages(X, cluster_assignments, self.n_clusters, 10)
        return self
    
    def fit_plsr(self,X, Y):
        self.plsr = PLSRegression(n_components = self.n_components).fit(self.transform(X),Y)
        return self
    
    def score(self, X, Y):
        print("Sc", X)
        R2Y = self.fit_plsr(self.transform(X, Y),Y).score(self.transform(X, Y),Y)
        y_pred = cross_val_predict(self.plsr, self.X_Filt_Clust_Avgs, Y, cv=self.Y.size)
        return R2Y, explained_variance_score(Y, y_pred) #Q2Y    
    
    def Scores_Loadings(self, X, Y):
        X_scores, Y_scores = self.fit_plsr(self.transform(X),Y).fit_transform(self.transform(X),Y)
        PC1_scores, PC2_scores = X_scores[:,0], X_scores[:,1]
        PC1_xload, PC2_xload = plsr.x_loadings_[:,0], plsr.x_loadings_[:,1]
        PC1_yload, PC2_yload = plsr.y_loadings_[:,0], plsr.y_loadings_[:,1]
        return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload    

    
###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def TunningHyperpar(X,Y):
    estimator = MyOwnKMEANS()
    param_grid = { 'n_clusters': np.arange(3,11), 'n_components': np.arange(3,11)}
    grid = GridSearchCV(estimator, param_grid=param_grid, cv=X.shape[0])
    fit = grid.fit(X, Y)
    CVresuts = pd.DataFrame(data=fit.cv_results_)
    return CVresults



# def GridSearch_nClusters(X):
#     kmeans = KMeans(init="k-means++")
#     parameters = {'n_clusters': np.arange(2,16)}
#     grid = GridSearchCV(kmeans, parameters, cv=X.shape[1])
#     fit = grid.fit(np.transpose(X))
#     CVresults_max = pd.DataFrame(data=fit.cv_results_)
#     std_scores = { '#Clusters': CVresults_max['param_n_clusters'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
#     CVresults_min = pd.DataFrame(data=std_scores)
#     return CVresults_max, CVresults_min