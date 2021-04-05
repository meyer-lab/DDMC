"""
This creates Figure 3: ABL/SFK/YAP experimental validations
"""

import pandas as pd
import numpy as np
<<<<<<< HEAD
import scipy as sp
from .common import subplotLabel, getSetup
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ..plsr import R2Y_across_components
from .figure1 import pca_dfs
from ..distances import DataFrameRipleysK
import matplotlib.colors as colors
from sklearn.model_selection import cross_val_predict
import matplotlib.cm as cm
=======
import matplotlib.pyplot as plt
>>>>>>> master
import seaborn as sns
from .common import subplotLabel, getSetup
from .figure1 import TimePointFoldChange

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 12), (3, 4), multz={8:1, 10:1})

    # Add subplot labels
    subplotLabel(ax)

<<<<<<< HEAD
    return f


def plotPCA(ax, d, n_components, scores_ind, loadings_ind, hue_scores=None, style_scores=None, pvals=None, style_load=None, legendOut=False):
    """ Plot PCA scores and loadings. """
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.select_dtypes(include=["float64"]).values)
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, scores_ind, loadings_ind)
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    sns.scatterplot(x="PC1", y="PC2", data=dScor_, hue=hue_scores, style=style_scores, ax=ax[0], **{"linewidth": 0.5, "edgecolor": "k"})
    ax[0].set_title("PCA Scores")
    ax[0].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[0].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    if legendOut:
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2)

    # Loadings
    if isinstance(pvals, np.ndarray):
        dLoad_["p-value"] = pvals
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="p-value", style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})
    else:
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})

    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    for j, txt in enumerate(dLoad_[loadings_ind]):
        ax[1].annotate(txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10)


def plotGridSearch(ax, gs):
    """ Plot gridsearch results by ranking. """
    ax = sns.barplot(x="rank_test_score", y="mean_test_score", data=np.abs(gs.iloc[:20, :]), ax=ax, **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax.set_title("Hyperaparameter Search")
    ax.set_xticklabels(np.arange(1, 21))
    ax.set_ylabel("Mean Squared Error")


def plotR2YQ2Y(ax, model, X, Y, b=3, color="darkblue"):
    """ Plot R2Y/Q2Y variance explained by each component. """
    Q2Y = R2Y_across_components(model, X, Y, b, crossval=True)
    R2Y = R2Y_across_components(model, X, Y, b)

    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align="center", label="Q2Y", color=color)
    ax.bar(range_ - 0.15, R2Y, width=0.3, align="center", label="R2Y", color="black")
    ax.set_title("R2Y/Q2Y - Cross-validation", fontsize=12)
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Variance", fontsize=11)
    ax.legend(loc=0)


def plotActualVsPredicted(ax, plsr_model, X, Y, y_pred="cross-validation", color="darkblue"):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    if y_pred == "cross-validation":
        Y_predictions = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
        ylabel = "Predicted"
    if y_pred == "fit":
        Y_predictions = plsr_model.fit(X, Y).predict(X)
        ylabel = "Fit"

    if len(Y.columns) > 1:
        for i, label in enumerate(Y.columns):
            y = Y.iloc[:, i]
            ypred = Y_predictions[:, i]
            ax[i].scatter(y, ypred, color=color)
            ax[i].plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
            ax[i].set_xlabel("Actual")
            ax[i].set_ylabel(ylabel)
            ax[i].set_title(label)

            ax[i].set_aspect("equal", "datalim")

            # Add correlation coefficient
            coeff, _ = sp.stats.pearsonr(ypred, y)
            textstr = "$r$ = " + str(np.round(coeff, 4))
            props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
            ax[i].text(0.75, 0.10, textstr, transform=ax[i].transAxes, verticalalignment="top", bbox=props)

    elif len(Y.columns) == 1:
        y = Y.iloc[:, 0]
        ypred = Y_predictions[:, 0]
        ax.scatter(y, ypred)
        ax.plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
        ax.set_xlabel("Actual")
        ax.set_ylabel(ylabel)
        ax.set_title(Y.columns[0])

        # Add correlation coefficient
        coeff, _ = sp.stats.pearsonr(ypred, y)
        textstr = "$r$ = " + str(np.round(coeff, 4))
        props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
        ax.text(0.75, 0.10, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, pcX=1, pcY=2, data="clusters", annotate=True):
    """Plot Scores and Loadings of PLSR model"""
    X_scores, _ = model.transform(X, Y)
    PC1_xload, PC2_xload = model.x_loadings_[:, pcX - 1], model.x_loadings_[:, pcY - 1]
    PC1_yload, PC2_yload = model.y_loadings_[:, pcX - 1], model.y_loadings_[:, pcY - 1]

    PC1_scores, PC2_scores = X_scores[:, pcX - 1], X_scores[:, pcY - 1]

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title("PLSR Model Scores", fontsize=12)
    ax[0].set_xlabel("Principal Component 1", fontsize=11)
    ax[0].set_ylabel("Principal Component 2", fontsize=11)
    ax[0].axhline(y=0, color="0.25", linestyle="--")
    ax[0].axvline(x=0, color="0.25", linestyle="--")

    spacer = 0.5
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
            ax[1].annotate(txt, (PC1_xload[i], PC2_xload[i]))
    markers = ["x", "D", "*", "1"]
    for i, label in enumerate(Y.columns):
        ax[1].annotate(label, (PC1_yload[i] + 0.001, PC2_yload[i] - 0.001))
        ax[1].scatter(PC1_yload[i], PC2_yload[i], color="black", marker=markers[i])
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].set_title("PLSR Model Loadings", fontsize=12)
    ax[1].set_xlabel("Principal Component 1", fontsize=11)
    ax[1].set_ylabel("Principal Component 2", fontsize=11)
    ax[1].axhline(y=0, color="0.25", linestyle="--")
    ax[1].axvline(x=0, color="0.25", linestyle="--")


def plotCenters(ax, centers, xlabels, title, yaxis=False):
    centers = pd.DataFrame(centers).T
    centers.columns = xlabels
    for i in range(centers.shape[0]):
        cl = pd.DataFrame(centers.iloc[i, :]).T
        m = pd.melt(cl, value_vars=list(cl.columns), value_name="p-signal", var_name="Lines")
        m["p-signal"] = m["p-signal"].astype("float64")
        sns.lineplot(x="Lines", y="p-signal", data=m, color="#658cbb", ax=ax, linewidth=2)
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_ylabel("$log_{10}$ p-signal")
        ax.xaxis.set_tick_params(bottom=True)
        ax.set_xlabel("")
        if title:
            ax.legend([title])
        else:
            ax.legend(["cluster " + str(i + 1)])
        if yaxis:
            ax.set_ylim([yaxis[0], yaxis[1]])


def plotMotifs(pssms, axes, titles=False, yaxis=False):
    """Generate logo plots of a list of PSSMs"""
    for i, ax in enumerate(axes):
        pssm = pssms[i].T
        pssm.index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        logo = lm.Logo(pssm,
                       font_name='Arial',
                       vpad=0.1,
                       width=.8,
                       flip_below=False,
                       center_values=False,
                       ax=ax)
        logo.ax.set_ylabel('information (bits)')
        logo.style_xticks(anchor=1, spacing=1)
        if titles:
            logo.ax.set_title(titles[i] + " Motif")
        else:
            logo.ax.set_title('Motif Cluster ' + str(i + 1))
        if yaxis:
            logo.ax.set_ylim(yaxis)


def plot_LassoCoef(ax, model, title=False):
    """Plot Lasso Coefficients"""
    coefs = pd.DataFrame(model.coef_).T
    coefs.index += 1
    coefs = coefs.reset_index()
    coefs.columns = ["Cluster", 'Viability', 'Apoptosis', 'Migration', 'Island']
    m = pd.melt(coefs, id_vars="Cluster", value_vars=list(coefs.columns)[1:], var_name="Phenotype", value_name="Coefficient")
    sns.barplot(x="Cluster", y="Coefficient", hue="Phenotype", data=m, ax=ax)
    if title:
        ax.set_title(title)
=======
    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
>>>>>>> master

    # Dasatinib Dose Response Time Course
    das = [pd.read_csv("msresist/data/Validations/Dasatinib.csv"), pd.read_csv("msresist/data/Validations/Dasatinib_2fixed.csv")]
    das = transform_YAPviability_data(das)
    plot_YAPinhibitorTimeLapse(ax[:8], das, ylim=[0, 14])

    # YAP blot AXL vs KO
    ax[8].axis("off")

    # YAP blot dasatinib dose response
    ax[9].axis("off")

<<<<<<< HEAD
def plotUpstreamKinase_heatmap(model, clusters, ax):
    """Plot Frobenius norm between kinase PSPL and cluster PSSMs"""
    ukin = model.predict_UpstreamKinases()
    ukin_mc = MeanCenter(ukin, mc_col=True, mc_row=True)
    ukin_mc.columns = ["Kinase"] + list(np.arange(1, model.ncl + 1))
    data = ukin_mc.sort_values(by="Kinase").set_index("Kinase")[clusters]
    sns.heatmap(data.T, ax=ax, xticklabels=data.index, cbar_kws={"shrink": 0.75})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    ax.set_ylabel("Frobenius Norm (motif vs kinase specifcity)")
    ax.set_title("Upstream Kinase Inference")
=======
    return f
>>>>>>> master


def plot_YAPinhibitorTimeLapse(ax, X, ylim=False):
    lines = ["WT", "KO"]
    treatments = ["UT", "E", "E/R", "E/A"]
    for i, line in enumerate(lines):
        for j, treatment in enumerate(treatments):
            if i > 0:
                j += 4
            m = X[X["Lines"] == line]
            m = m[m["Condition"] == treatment]
            sns.lineplot(x="Elapsed", y="Fold-change confluency", hue="Inh_concentration", data=m, ci=68, ax=ax[j])
            ax[j].set_title(line + "-" + treatment)
            if ylim:
                ax[j].set_ylim(ylim)
            if i != 0 or j != 0:
                ax[j].get_legend().remove()
            else:
                ax[j].legend(prop={'size':10})


def transform_YAPviability_data(data, itp=12):
    """Transform to initial time point and convert into seaborn format"""
    new = []
    for i, mat in enumerate(data):
        if i > 0:
            mat = MeanTRs(mat)
        new.append(TimePointFoldChange(mat, itp))

    c = pd.concat(new, axis=0)
    c = pd.melt(c, id_vars="Elapsed", value_vars=c.columns[1:], var_name="Lines", value_name="Fold-change confluency")
    c["Condition"] = [s.split(" ")[1].split(" ")[0] for s in c["Lines"]]
    c["Inh_concentration"] = [s[4:].split(" ")[1] for s in c["Lines"]]
    c["Lines"] = [s.split(" ")[0] for s in c["Lines"]]
    c = c[["Elapsed", "Lines", "Condition", "Inh_concentration", "Fold-change confluency"]]
    c = c[c["Elapsed"] >= itp]
    return c


def MeanTRs(X):
    """Merge technical replicates of 2 BR by taking the mean."""
    idx = [np.arange(0, 6) + i for i in range(1, X.shape[1], 12)]
    for i in idx:
        for j in i:
            X.iloc[:, j] = X.iloc[:, [j, j + 6]].mean(axis=1)
            X.drop(X.columns[j + 6], axis="columns")

    return X.drop(X.columns[[j + 6 for i in idx for j in i]], axis="columns")
