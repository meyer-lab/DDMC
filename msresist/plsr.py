"""PLSR analysis functions (plotting functions are located in msresist/figures/figure2)"""

import numpy as np
import seaborn as sns
import scipy as sp
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
import matplotlib.colors as colors
import matplotlib.cm as cm

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


def plotStripActualVsPred(ax, n_components, Xs, Y, models):
    """Actual vs Predicted of different PLSR models"""
    datas = []
    for ii, X in enumerate(Xs):
        data = pd.DataFrame()
        Y_predictions = cross_val_predict(PLSRegression(n_components=n_components[ii]), X, Y, cv=Y.shape[0])
        coeff = [sp.stats.pearsonr(Y_predictions[:, jj], Y.iloc[:, jj])[0] for jj in range(len(Y.columns))]
        data["Phenotype"] = list(Y.columns)
        data["r-score"] = coeff
        data["Model"] = models[ii]
        datas.append(data)
    res = pd.concat(datas)
    sns.stripplot(x="Phenotype", y="r-score", data=res, ax=ax, hue="Model")
    ax.set_title("Actual vs Predicted")
    ax.legend(prop={'size': 8})


def plotActualVsPredicted(ax, plsr_model, X, Y, y_pred="cross-validation", color="darkblue", type="scatter", title=False):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    if y_pred == "cross-validation":
        Y_predictions = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
        ylabel = "Predicted"
    if y_pred == "fit":
        Y_predictions = plsr_model.fit(X, Y).predict(X)
        ylabel = "Fit"

    if len(Y.columns) > 1:
        if type == "scatter":
            for i, label in enumerate(Y.columns):
                y = Y.iloc[:, i]
                ypred = Y_predictions[:, i]
                ax[i].scatter(y, ypred, color=color)
                ax[i].plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
                ax[i].set_xlabel("Actual")
                ax[i].set_ylabel(ylabel)
                ax[i].set_title(label)
                ax[i].set_aspect("equal", "datalim")
                add_rBox(ypred, y, ax)

        elif type == "bar":
            coeff = [sp.stats.pearsonr(Y_predictions[:, i], Y.iloc[:, i])[0] for i in range(len(Y.columns))]
            data = pd.DataFrame()
            data["Phenotype"] = list(Y.columns)
            data["r-score"] = coeff
            sns.barplot(x="Phenotype", y="r-score", data=data, ax=ax, color=color, **{"linewidth": 0.5}, **{"edgecolor": "black"})
            if title:
                ax.set_title(title)

    elif len(Y.columns) == 1:
        y = Y.iloc[:, 0]
        ypred = Y_predictions[:, 0]
        ax.scatter(y, ypred)
        ax.plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
        ax.set_xlabel("Actual")
        ax.set_ylabel(ylabel)
        ax.set_title(Y.columns[0])
        add_rBox(ypred, y, ax)


def add_rBox(ypred, y, ax):
    """Add correlation coefficient box onto scatter plot of Actual vs Predicted."""
    coeff, _ = sp.stats.pearsonr(ypred, y)
    textstr = "$r$ = " + str(np.round(coeff, 4))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax.text(0.75, 0.10, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, pcX=1, pcY=2, data="clusters", annotate=True, spacer=0.05):
    """Plot Scores and Loadings of PLSR model"""
    X_scores, _ = model.transform(X, Y)
    PC1_xload, PC2_xload = model.x_loadings_[:, pcX - 1], model.x_loadings_[:, pcY - 1]
    PC1_yload, PC2_yload = model.y_loadings_[:, pcX - 1], model.y_loadings_[:, pcY - 1]

    PC1_scores, PC2_scores = X_scores[:, pcX - 1], X_scores[:, pcY - 1]

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]), fontsize=10)
    ax[0].set_title("PLSR Model Scores", fontsize=12)
    ax[0].set_xlabel("Principal Component 1", fontsize=11)
    ax[0].set_ylabel("Principal Component 2", fontsize=11)
    ax[0].axhline(y=0, color="0.25", linestyle="--")
    ax[0].axvline(x=0, color="0.25", linestyle="--")

    ax[0].set_xlim([(-1 * max(np.abs(PC1_scores))) - spacer, max(np.abs(PC1_scores)) + spacer])
    ax[0].set_ylim([(-1 * max(np.abs(PC2_scores))) - spacer, max(np.abs(PC2_scores)) + spacer])

    # Loadings
    if data != "clusters":
        ncl = X.shape[1]

    colors_ = cm.rainbow(np.linspace(0, 1, ncl))
    if annotate:
        numbered = []
        list(map(lambda v: numbered.append(str(v + 1)), range(ncl)))
        for i, txt in enumerate(numbered):
            ax[1].annotate(txt, (PC1_xload[i] + spacer, PC2_xload[i] - spacer), fontsize=10)
    markers = ["x", "D", "*", "1"]
    for i, label in enumerate(Y.columns):
        ax[1].annotate(label, (PC1_yload[i] + spacer, PC2_yload[i] - spacer), fontsize=10)
        ax[1].scatter(PC1_yload[i], PC2_yload[i], color="black", marker=markers[i])
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].set_title("PLSR Model Loadings", fontsize=12)
    ax[1].set_xlabel("Principal Component 1", fontsize=11)
    ax[1].set_ylabel("Principal Component 2", fontsize=11)
    ax[1].axhline(y=0, color="0.25", linestyle="--")
    ax[1].axvline(x=0, color="0.25", linestyle="--")