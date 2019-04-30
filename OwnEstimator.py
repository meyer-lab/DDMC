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
    
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
    
    def fit(self, X, Y):    
        print("kmfit", X)
        X, Y = check_X_y(X, Y)                  
        self.kmeans = KMeans(n_clusters=self.n_clusters).fit(np.transpose(X))
        self.X_ = X
        self.Y_ = Y
        return self
        
    def transform(self,X):
        check_is_fitted(self, ['X_', 'Y_'])
        X = check_array(X)
        print("trans", X)
        cluster_assignments = self.kmeans.predict(np.transpose(X))
        X_Filt_Clust_Avgs = ClusterAverages(X, cluster_assignments, self.n_clusters, 10)
        print("X_Filt_Clust_Avgs", X_Filt_Clust_Avgs)
        return X_Filt_Clust_Avgs
    
class MyOwnRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components = 2):
        self.n_components = n_components
    
    def fit(self, X, Y):
        print("PLSRfit", X, "\n", Y)
        self.plsr = PLSRegression(n_components = self.n_components).fit(X,Y)      
        print("check")
        return self
    
    def R2YQ2Y(self,X,Y):
        R2Y = self.plsr.score(X,Y)
        y_pred = cross_val_predict(self.plsr, X, Y, cv=self.Y.size)
        return R2Y, explained_variance_score(Y, y_pred)

#     def Scores_Loadings(self, X, Y):
#         X_scores, Y_scores = self.plsr.fit_transform(X,Y)
#         PC1_scores, PC2_scores = X_scores[:,0], X_scores[:,1]
#         PC1_xload, PC2_xload = plsr.x_loadings_[:,0], plsr.x_loadings_[:,1]
#         PC1_yload, PC2_yload = plsr.y_loadings_[:,0], plsr.y_loadings_[:,1]
#         return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload
    
###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def TunningHyperpar(X,Y):
    estimators = [('kmeans', MyOwnKMEANS()), ('plsr', MyOwnRegressor())]
    pipe = Pipeline(estimators)
    param_grid = dict(kmeans__n_clusters = np.arange(2,11), plsr__n_components = np.arange(2,11))
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=X.shape[0])
    fit = grid.fit(X, Y)
    CVresuts = pd.DataFrame(data=fit.cv_results_)
    return CVresults
