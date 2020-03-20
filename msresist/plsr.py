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
<<<<<<< HEAD
        plsr_model = PLSRegression(n_components=b)
        y_pred = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
=======
        if cv == 1:
            model.set_params(n_components=b)
        if cv == 2:
            model.set_params(plsr__n_components=b)
        y_pred = cross_val_predict(model, X, Y, cv=Y.size, n_jobs=-1)
>>>>>>> master
        Q2Ys.append(explained_variance_score(Y, y_pred))
    return Q2Ys


def PLSR(X, Y, nComponents):
    """ Run PLSR. """
    plsr = PLSRegression(n_components=nComponents)
    X_scores, _ = plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = plsr.x_loadings_[:, 0], plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = plsr.y_loadings_[:, 0], plsr.y_loadings_[:, 1]
    return plsr, PC1_scores, PC2_scores, PC1_xload, PC2_xload, PC1_yload, PC2_yload
