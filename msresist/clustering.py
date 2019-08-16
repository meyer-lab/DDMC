"Clustering functions"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
    

class MyOwnKMEANS(BaseEstimator):
    """ Runs k-means providing the centers and cluster members and sequences """

    def __init__(self, n_clusters):
        """ define variables """
        self.n_clusters = n_clusters

    def fit(self, X, Y):
        """ fit data into k-means """
        self.kmeans_ = KMeans(n_clusters=self.n_clusters).fit(X.T)
        return self

    def transform(self, X):
        """ calculate cluster averages """
        centers, _ = ClusterAverages(X, self.kmeans_.labels_)
        return centers

    def clustermembers(self, X):
        """ generate dictionary containing peptide names and sequences for each cluster """
        _, clustermembers = ClusterAverages(X, self.kmeans_.labels_)
        return clustermembers


###------------ Computing Cluster Averages / Identifying Cluster Members ------------------###
    
def ClusterAverages(X, labels):
    "calculate cluster averages and dictionary with cluster members and sequences"
    X = X.T.assign(cluster = labels)
    centers = []
    dict_clustermembers = {}
    for i in range(0, max(labels)+1):
        centers.append(list(X[X["cluster"] == i].iloc[:, :-1].mean()))
        dict_clustermembers["Cluster_" + str(i+1)] = list(X[X["cluster"] == i].iloc[:, 1])
        dict_clustermembers["seqs_Cluster_" + str(i+1)] = list(X[X["cluster"] == i].iloc[:, 0])

    return pd.DataFrame(centers).T, pd.DataFrame(dict([(k, pd.Series(v)) for k,v in dict_clustermembers.items()]))