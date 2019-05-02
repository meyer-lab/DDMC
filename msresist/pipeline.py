import numpy as np, pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin, ClusterMixin, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from .plsr import FilteringOutPeptides, ClusterAverages
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV


###------------ Creating Own Estimators ------------------###
'''
Unresolved issues / questions: 
    - Averaged Clusters well generated but generated incompatibility in def score(). X is matrix cond: #clus an
'''

class MyOwnKMEANS(BaseEstimator):
    
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
    
    def fit(self, X, Y):
        #print("fit", X)
        self.cluster_assignments_ = KMeans(n_clusters=self.n_clusters).fit_predict(np.transpose(X))

        self.X_ = X
        self.Y_ = Y
        return self
        
    def transform(self,X):
        check_is_fitted(self, ['X_', 'Y_'])
        X = check_array(X)
#         print("trans", X)
#         print("trans shape", X.shape)
#         print("cluster assignments:", self.cluster_assignments_) 
        X_Filt_Clust_Avgs = ClusterAverages(X, self.cluster_assignments_, self.n_clusters, X.shape[0])
#         print("clust", self.n_clusters)
#         print("X_Filt_Clust_Avgs", X_Filt_Clust_Avgs)
        return X_Filt_Clust_Avgs
    
class MyOwnRegressor(BaseEstimator):
    def __init__(self, n_components):
        self.n_components = n_components
        pass
    
    def fit(self, X, Y):    
        X = np.squeeze(np.array(X))
#         print("plsrfit", X, "\n", Y)
        self.plsr = PLSRegression(n_components = self.n_components).fit(X,Y)
#         print("check", X, "\n", Y)
        return self
    
    def score(self,X,Y):
#         X = np.reshape(X, (1, X.size))
#         print("ncomp:", self.n_components)
#         print("R2YQ2Y:", X, Y)       #X(1, 96) Y(1,)
        R2Y = self.plsr.score(X,Y)
#         print("R2Y:", R2Y)
#         y_pred = cross_val_predict(self.plsr, X, Y, cv=self.Y.size)
        return R2Y #explained_variance_score(Y, y_pred)

    def Scores_Loadings(self, X, Y):
        X_scores, Y_scores = self.plsr.transform(X,Y)
        PC1_scores, PC2_scores = X_scores[:,0], X_scores[:,1]
        PC1_xload, PC2_xload = self.plsr.x_loadings_[:,0], plsr.x_loadings_[:,1]
        PC1_yload, PC2_yload = self.plsr.y_loadings_[:,0], plsr.y_loadings_[:,1]
        return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload
    
###------------ Building Pipeline and Tunning Hyperparameters ------------------###

def TunningHyperpar(X,Y):
    #estimators = [('kmeans', MyOwnKMEANS()), ('plsr', PLSRegression())]
    estimators = [('kmeans', MyOwnKMEANS(5)), ('plsr', MyOwnRegressor(2))]
    pipe = Pipeline(estimators)
    param_grid = dict(kmeans__n_clusters = np.arange(5,16), plsr__n_components = np.arange(2,6))
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=X.shape[0])
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = { '#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min