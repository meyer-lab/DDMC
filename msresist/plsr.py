"""PLSR analysis functions (plotting functions are located in msresist/figures/figure2)"""

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score

###------------ PLSR model functions ------------------###


def R2Y_across_components(model, X, Y, max_comps, crossval=False):
    """ Calculate R2Y or Q2Y, depending upon crossval. """
    R2Ys = []

    for b in range(1, max_comps):
        model.set_params(n_components=b)
        if crossval is True:
            y_pred = cross_val_predict(model, X, Y, cv=Y.shape[0])
        else:
            y_pred = model.fit(X, Y).predict(X)

        R2Ys.append(explained_variance_score(Y, y_pred))
    return R2Ys
