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


def plotMeasuredVsPredicted(ax, plsr_model, X, Y):
    "Plot exprimentally-measured vs PLSR-predicted values"
    Y_predictions = np.squeeze(cross_val_predict(plsr_model, X, Y, cv=Y.size))
    ax.scatter(Y, np.squeeze(Y_predictions))
    ax.plot(np.unique(Y), np.poly1d(np.polyfit(Y, np.squeeze(Y_predictions), 1))(np.unique(Y)), color="r")
    ax.set(title="Correlation Measured vs Predicted", xlabel="Actual Y", ylabel="Predicted Y")
    ax.set_title("Correlation Measured vs Predicted")
    ax.set_xlabel("Measured Cell Viability")
    ax.set_ylabel("Predicted Cell Viability")
    coeff, pval = sp.stats.pearsonr(list(Y_predictions), list(Y))
    textstr = "$r$ = " + str(np.round(coeff, 4))
    props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
    ax.text(0.80, 0.09, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props);


""" Avoid data labels from overlapping. """
# source: https://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 1.5: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions


def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - .03, 1.02*t, '%d'%int(y),rotation=0, color='blue', fontsize=13)
        if y != t:
            axis.arrow(x, t+20,0,y-t, color='blue',alpha=0.2, width=txt_width*0.0,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)