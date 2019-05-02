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
    - Q2Y method, currently only R2Y?
    - Fit only X_train and y_train? 
    - Error in Scores_Loadings: operands could not be broadcast together with shapes (10,5) (96,) (10,5). self.plsr_ not correcly imported?
    - How does GridsearchCV's CV work? Way to get a score for the KMeans alone? 
'''

class MyEstimator(BaseEstimator):
    
    def __init__(self, n_clusters, n_components):
        self.kmeans_ = KMeans(n_clusters=n_clusters)
        self.plsr_ = PLSRegression(n_components=n_components)
    
    def fit(self, X, Y):
        self.kmeans_.fit(np.transpose(X))
        
        # Cluster centers are the averages, per definition of kmeans
        centers = np.array(self.kmeans_.cluster_centers_)
        
        # Fit PLSR model. Result saved in the PLSR class.
        self.plsr_.fit(centers.T, Y)
    
    def predict(self, X):
        # TODO: Should add assertions about the expected size of X, based upon training
        clustPred = ClusterAverages(X, self.kmeans_.labels_)
        # Not sure this is what the function handles?
    
    def transform(self, X):
        check_is_fitted(self, ['X_', 'Y_'])
        X = check_array(X)
        cluster_assignments_ = self.kmeans_.predict(np.transpose(X))
        X_Filt_Clust_Avgs = ClusterAverages(X, cluster_assignments_, self.n_clusters, 10)
        print(X_Filt_Clust_Avgs)
        raise SystemExit
        return X
    
    def score(self,X,Y):
        print(X.shape)
        print(Y.shape)
        raise  SystemExit
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
