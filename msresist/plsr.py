"PLSR functions"

import scipy as sp
from scipy.stats import zscore
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import explained_variance_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV


###------------ Scaling Matrices ------------------###

def zscore_columns(matrix):
    """ Z-score each column of the matrix. Note that
    sklearn PLSRegression already handles scaling. """
    return zscore(matrix, axis=0)


###------------ Q2Y/R2Y ------------------###

def R2Y_across_components(X, Y, max_comps):
    "Calculate R2Y."
    R2Ys = []
    for b in range(1, max_comps):
        plsr = PLSRegression(n_components=b)
        plsr.fit(X, Y)
        R2Y = plsr.score(X, Y)
        R2Ys.append(R2Y)
    return R2Ys


def Q2Y_across_components(X, Y, max_comps):
    "Calculate Q2Y using cros_val_predct method."
    Q2Ys = []
    for b in range(1, max_comps):
        plsr_model = PLSRegression(n_components=b)
        y_pred = cross_val_predict(plsr_model, X, Y, cv=Y.size)
        Q2Ys.append(explained_variance_score(Y, y_pred))
    return Q2Ys


def Q2Y_across_comp_manual(X_z, Y_z, max_comps, sublabel):
    "Calculate Q2Y manually."
    PRESS = 0
    SS = 0
    Q2Ys = []
    for b in range(1, max_comps):
        plsr_model = PLSRegression(n_components=b)
        for train_index, test_index in LeaveOneOut().split(X_z, Y_z):
            X_train, X_test = X_z[train_index], X_z[test_index]
            Y_train, Y_test = Y_z[train_index], Y_z[test_index]
            X_train = zscore_columns(X_train)
            Y_train = sp.stats.zscore(Y_train)
            plsr_model.fit_transform(X_train, Y_train)
            Y_predict_cv = plsr_model.predict(X_test)
            PRESS_i = (Y_predict_cv - Y_test) ** 2
            SS_i = (Y_test) ** 2
            PRESS = PRESS + PRESS_i
            SS = SS + SS_i
        Q2Y = 1 - (PRESS / SS)
        Q2Ys.append(Q2Y)
    return Q2Ys

###------------ Fitting PLSR and CV ------------------###


def PLSR(X, Y, nComponents):
    "Run PLSR."
    plsr = PLSRegression(n_components=nComponents)
    X_scores, Y_scores = plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = plsr.x_loadings_[:, 0], plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = plsr.y_loadings_[:, 0], plsr.y_loadings_[:, 1]
    return plsr, PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload


def MeasuredVsPredicted_LOOCVplot(X, Y, plsr_model, fig, ax, axs):
    "Plot exprimentally-measured vs PLSR-predicted values"
    Y_predictions = np.squeeze(cross_val_predict(plsr_model, X, Y, cv=Y.size))
    coeff, pval = sp.stats.pearsonr(list(Y_predictions), list(Y))
    print("Pearson's R: ", coeff, "\n", "p-value: ", pval)
    if ax == "none":
        plt.scatter(Y, np.squeeze(Y_predictions))
        plt.plot(np.unique(Y), np.poly1d(np.polyfit(Y, np.squeeze(Y_predictions), 1))(np.unique(Y)), color = "r")
        plt.title("Correlation Measured vs Predicted")
        plt.xlabel("Measured Cell Viability")
        plt.ylabel("Predicted Cell Viability")
    else:
        axs[ax].scatter(Y, np.squeeze(Y_predictions))
        axs[ax].set(title="Correlation Measured vs Predicted", xlabel='Actual Y', ylabel='Predicted Y')

###------------ Computing Cluster Averages ------------------###

def ClusterAverages(X_, cluster_assignments, nClusters, nObs, ProtNames, peptide_phosphosite):
    "calculate cluster averages and dictionary with cluster members and sequences"
    X_FCl = np.insert(X_, 0, cluster_assignments, axis=0)  # 11:96   11 = 10cond + clust_assgms
    X_FCl = np.transpose(X_FCl)  # 96:11
    ClusterAvgs_arr = np.zeros((nClusters, nObs))  # 5:10
    DictClusterToMembers = {}
    for i in range(nClusters):
        CurrentCluster = []
#         DictMemberToSeq = {}
        ClusterMembers = []
        ClusterSeqs = []
        for idx, arr in enumerate(X_FCl):
            if i == arr[0]:  # arr[0] is the location of the cluster assignment of the specific peptide
                CurrentCluster.append(arr)  # array with 96:11, so every arr contains a single peptide's values
                ClusterMembers.append(ProtNames[idx])
                ClusterSeqs.append(peptide_phosphosite[idx])
#                 DictMemberToSeq[ProtNames[idx]] = peptide_phosphosite[idx]
        CurrentCluster_T = np.transpose(CurrentCluster)  # 11:96, so every arr contains a single condition's values (eg. all peptides values within cluster X in Erl)
        CurrentAvgs = []
        for x, arr in enumerate(CurrentCluster_T):
            if x == 0:  # cluster assignments
                continue
            else:
                avg = np.mean(arr)
                CurrentAvgs.append(avg)
        DictClusterToMembers[i + 1] = (ClusterMembers)
        DictClusterToMembers["Seqs_Cluster_" + str(i + 1)] = ClusterSeqs
        ClusterAvgs_arr[i, :] = CurrentAvgs
        AvgsArr = np.transpose(ClusterAvgs_arr)
    return AvgsArr, DictClusterToMembers


def GridSearch_CV(model, X, Y, parameters, cv, scoring=None):
    """ Exhaustive search over specified parameter values for an estimator. """
    grid = GridSearchCV(model, param_grid=parameters, cv=cv, scoring=scoring)
    fit = grid.fit(X, Y)
    CVresults_max = pd.DataFrame(data=fit.cv_results_)
    return CVresults_max
