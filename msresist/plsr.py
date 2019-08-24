"""PLSR functions"""

import scipy as sp
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn.cross_decomposition import PLSRegression


def zscore_columns(matrix):
    """ Z-score each column of the matrix. Note that
    sklearn PLSRegression already handles scaling. """
    return zscore(matrix, axis=0)


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


def PLSR(X, Y, nComponents):
    "Run PLSR."
    plsr = PLSRegression(n_components=nComponents)
    X_scores, _ = plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = plsr.x_loadings_[:, 0], plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = plsr.y_loadings_[:, 0], plsr.y_loadings_[:, 1]
    return plsr, PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload


def MeasuredVsPredicted_LOOCVplot(X, Y, plsr_model, _, ax, axs):
    "Plot exprimentally-measured vs PLSR-predicted values"
    Y_predictions = np.squeeze(cross_val_predict(plsr_model, X, Y, cv=Y.size))
    coeff, pval = sp.stats.pearsonr(list(Y_predictions), list(Y))
    print("Pearson's R: ", coeff, "\n", "p-value: ", pval)
    if ax == "none":
        plt.scatter(Y, np.squeeze(Y_predictions))
        plt.plot(np.unique(Y), np.poly1d(np.polyfit(Y, np.squeeze(Y_predictions), 1))(np.unique(Y)), color="r")
        plt.title("Correlation Measured vs Predicted")
        plt.xlabel("Measured Cell Viability")
        plt.ylabel("Predicted Cell Viability")
    else:
        axs[ax].scatter(Y, np.squeeze(Y_predictions))
        axs[ax].set(title="Correlation Measured vs Predicted", xlabel="Actual Y", ylabel="Predicted Y")
