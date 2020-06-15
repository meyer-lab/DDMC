"""PLSR analysis functions (plotting functions are located in msresist/figures/figure2)"""

import scipy as sp
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import explained_variance_score

###------------ PLSR model functions ------------------###


def zscore_columns(matrix):
    """ Z-score each column of the matrix. Note that
    sklearn PLSRegression already handles scaling. """
    return zscore(matrix, axis=0)


def R2Y_across_components(model, X, Y, cv, max_comps):
    """ Calculate R2Y. """
    R2Ys = []
    for b in range(1, max_comps):
        if cv == 1:
            model.set_params(n_components=b)
        if cv == 2:
            model.set_params(plsr__n_components=b)
        model.fit(X, Y)
        R2Ys.append(model.score(X, Y))
    return R2Ys


def Q2Y_across_components(model, X, Y, cv, max_comps):
    """ Calculate Q2Y using cros_val_predct method. """
    Q2Ys = []
    for b in range(1, max_comps):
        if cv == 1:
            model.set_params(n_components=b)
        if cv == 2:
            model.set_params(plsr__n_components=b)
        y_pred = cross_val_predict(model, X, Y, cv=Y.shape[0], n_jobs=-1)
        Q2Ys.append(explained_variance_score(Y, y_pred))
    return Q2Ys


def Q2Y_across_comp_manual(model, X, Y, cv, max_comps):
    "Calculate Q2Y manually."
    PRESS = 0
    SS = 0
    Q2Ys = []
    cols = X.columns
    Y = np.array(Y)
    X = sp.stats.zscore(X, axis=0)
    for b in range(1, max_comps):
        # Cross-validation across fixed clusters
        if cv == 1:
            model.set_params(n_components=b)
            for train_index, test_index in LeaveOneOut().split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                Y_train = sp.stats.zscore(Y_train, axis=0)
                X_train = sp.stats.zscore(X_train, axis=0)
                model.fit(X_train, Y_train)
                Y_predict = model.predict(X_test)
                PRESS_i = (Y_predict - Y_test) ** 2
                SS_i = (Y_test) ** 2
                PRESS = np.mean(PRESS + PRESS_i)
                SS = np.mean(SS + SS_i)

        # Chain long cross-validation
        if cv == 2:
            model.set_params(plsr__n_components=b)
            for train_index, test_index in LeaveOneOut().split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                Y_train = sp.stats.zscore(Y_train)
                X_train = pd.DataFrame(X_train)
                X_train.columns = cols
                model.fit(pd.DataFrame(X_train), Y_train)
                Y_predict = model.predict(pd.DataFrame(X_test))
                PRESS_i = (Y_predict - Y_test) ** 2
                SS_i = (Y_test) ** 2
                PRESS = np.mean(PRESS + PRESS_i)
                SS = np.mean(SS + SS_i)
        Q2Y = 1 - (PRESS / SS)
        Q2Ys.append(Q2Y)
    return Q2Ys
