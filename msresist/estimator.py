import numpy as np
import pandas as pd
from msresist.plsr import ClusterAverages
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.utils import check_consistent_length
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted, check_array


###------------ Creating Own Estimators ------------------###
'''
Unresolved issues / questions:
    - R2Y method
'''

class kmeansPLSR(BaseEstimator):

    def __init__(self, n_clusters, n_components):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.kmeans_ = KMeans(n_clusters=n_clusters)
        self.plsr_ = PLSRegression(n_components=n_components)

    def fit(self, X, Y):
        assignments_ = self.kmeans_.fit_predict(np.transpose(X))
        self.centers_ = ClusterAverages(X, assignments_, self.n_clusters, X.shape[0])
#         self.centers_ = np.array(self.kmeans_.cluster_centers_).T    #cond(10):pept(96)
        self.plsr_.fit(self.centers_, Y)       #cond(9):clusters(2) 
        return self
    
    def predict(self, X):
        # TODO: Should add assertions about the expected size of X, based upon training
        y_pred = self.plsr_.predict(X)
        return y_pred
    
    def score(self, X, Y):   #self.predict only works if the shape of test is 9,2 (broadcast shape?)
        print(X)
        print(Y)
        print(self.centers_)
        print(self.plsr_.predict(np.reshape(self.centers_.T[0], (9,1))))
        currentX = np.reshape(self.centers_, (9,2))
        R2Y = self.plsr_.score(currentX,Y)
        print("R2Y")
        return R2Y
    
    def Scores_Loadings(self, X, Y):
        X_scores, Y_scores = self.plsr_.transform(X, Y)
        PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
        PC1_xload, PC2_xload = plsr.x_loadings_[:, 0], plsr.x_loadings_[:, 1]
        PC1_yload, PC2_yload = plsr.y_loadings_[:, 0], plsr.y_loadings_[:, 1]
        return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload

###-------------- Tunning Hyperparameters ------------------###


def TunningHyperpar(X, Y):
    parameters = {'n_clusters': np.arange(2, 11), 'n_components': np.arange(2, 11)}
    grid = GridSearchCV(kmeansPLSR(2, 2), parameters, cv=X.shape[0])
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    std_scores = {'#Clusters': CVresults_max['param_n_clusters'], '#Components': CVresults_max['param_n_components'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
    CVresults_min = pd.DataFrame(data=std_scores)
    return CVresults_max, CVresults_min
