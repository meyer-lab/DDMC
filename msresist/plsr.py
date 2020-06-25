"""PLSR analysis functions (plotting functions are located in msresist/figures/figure2)"""

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score

###------------ PLSR model functions ------------------###


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
