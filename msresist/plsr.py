"""PLSR analysis functions (plotting functions are located in msresist/figures/figure2)"""

from scipy.stats import zscore
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn.cross_decomposition import PLSRegression

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


def PLSR(X, Y, nComponents):
    """ Run PLSR. """
    plsr = PLSRegression(n_components=nComponents)
    X_scores, _ = plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = plsr.x_loadings_[:, 0], plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = plsr.y_loadings_[:, 0], plsr.y_loadings_[:, 1]
    return plsr, PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload
