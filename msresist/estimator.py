# import numpy as np
# import pandas as pd
# from msresist.plsr import ClusterAverages
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.cluster import KMeans
# from sklearn.base import BaseEstimator
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import explained_variance_score, r2_score
# from sklearn.utils import check_consistent_length
# from sklearn.model_selection import GridSearchCV
# from sklearn.utils.validation import check_is_fitted, check_array


# ###------------ Creating Own Estimators ------------------###

# class kmeansPLSR(BaseEstimator):
#     """ Currently not working, pipelined model in comp_estimator file """
#     def __init__(self, n_clusters, n_components, ProtNames, peptide_phosphosite):
#         self.n_clusters = n_clusters
#         self.n_components = n_components
#         self.ProtNames = ProtNames
#         self.peptide_phosphosite = peptide_phosphosite
#         self.kmeans_ = KMeans(n_clusters=n_clusters)
#         self.plsr_ = PLSRegression(n_components=n_components)

#     def fit(self, X, Y):
#         assignments_ = self.kmeans_.fit_predict(np.transpose(X))
#         self.centers_, self.DictClusterToMembers = ClusterAverages(X, assignments_, self.n_clusters, X.shape[0], self.ProtNames, self.peptide_phosphosite)
# #         self.centers_ = np.array(self.kmeans_.cluster_centers_).T    #cond(10):pept(96)
#         self.plsr_.fit(self.centers_, Y)  # cond(9):clusters(2)
#         return self

#     def predict(self, X):
#         # TODO: Should add assertions about the expected size of X, based upon training
#         y_pred = self.plsr_.predict(self.centers_)
#         return y_pred

#     def score(self, X, Y):
#         currentX = np.reshape(self.centers_[0, :], (1, 2))
#         R2Y = self.plsr_.score(np.reshape(self.centers_[0, :], (1, 2)), Y)
#         print(R2Y)
#         return R2Y

#     def Scores_Loadings(self, X, Y):
#         X_scores, Y_scores = self.plsr_.transform(X, Y)
#         PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
#         PC1_xload, PC2_xload = self.plsr_.x_loadings_[:, 0], self.plsr_.x_loadings_[:, 1]
#         PC1_yload, PC2_yload = self.plsr_.y_loadings_[:, 0], self.plsr_.y_loadings_[:, 1]
#         return PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload

# ###-------------- Tunning Hyperparameters ------------------###


# def TunningHyperpar(X, Y, ProtNames, peptide_phosphosite):
#     parameters = {'n_clusters': np.arange(2, 11), 'n_components': np.arange(2, 11)}
#     param_grid = []

#     for nn in range(2, 16):
#         param_grid.append(dict(n_clusters=[nn], n_components=np.arange(1, nn + 1)))

#     grid = GridSearchCV(kmeansPLSR(2, 1, ProtNames, peptide_phosphosite), param_grid=param_grid, cv=X.shape[0], return_train_score=True)
#     fit = grid.fit(X, Y)
#     CVresults_max = pd.DataFrame(data=fit.cv_results_)
#     std_scores = {'#Clusters': CVresults_max['param_kmeans__n_clusters'], '#Components': CVresults_max['param_plsr__n_components'], 'std_test_scores': CVresults_max["std_test_score"], 'std_train_scores': CVresults_max["std_train_score"]}
#     CVresults_min = pd.DataFrame(data=std_scores)
#     return CVresults_max, CVresults_min, fit.best_params_
