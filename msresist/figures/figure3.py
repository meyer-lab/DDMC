"""
This creates Figure 2.
"""
import os
import random
import pandas as pd
import numpy as np
import scipy as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .common import subplotLabel, getSetup
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from ..clustering import MassSpecClustering
from ..plsr import R2Y_across_components
from ..figures.figure1 import pca_dfs
from ..distances import DataFrameRipleysK
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.model_selection import cross_val_predict
import matplotlib.cm as cm
import seaborn as sns
from ..pre_processing import preprocessing, y_pre, FixColumnLabels
import warnings
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (4, 4))

    # blank out first axis for cartoon
    #     ax[0].axis('off')

    # -------- Import and Preprocess Signaling Data -------- #
    X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)

    d = X.select_dtypes(include=["float64"]).T
    i = X.select_dtypes(include=["object"])

    all_lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    mut_lines = all_lines[1:]
    g_lines = all_lines[2:]

    d.index = all_lines

    # -------- Cell Phenotypes -------- #
    # Cell Viability
    cv1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR1_Phase.csv")
    cv2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR2_Phase.csv")
    cv3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")
    cv4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")

    itp = 24
    ftp = 96

    cv = [cv1, cv2, cv3, cv4]
    cv = FixColumnLabels(cv)

    v_ut = y_pre(cv, "UT", ftp, "Viability", all_lines, itp=itp)
    v_e = y_pre(cv, "-E", ftp, "Viability", all_lines, itp=itp)
    v_ae = y_pre(cv, "A/E", ftp, "Viability", all_lines, itp=itp)

    # Cell Death
    red1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR1_RedCount.csv")
    red2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR2_RedCount.csv")
    red3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR3_RedCount.csv")
    red4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR4_RedCount.csv")
    red4.columns = red3.columns

    for jj in range(1, red1.columns.size):
        red1.iloc[:, jj] /= cv1.iloc[:, jj]
        red2.iloc[:, jj] /= cv2.iloc[:, jj]
        red3.iloc[:, jj] /= cv3.iloc[:, jj]
        red4.iloc[:, jj] /= cv4.iloc[:, jj]

    cD = [red1, red2, red3, red4]
    cD = FixColumnLabels(cD)
    cd_ut = y_pre(cD, "UT", ftp, "Apoptosis", all_lines, itp=itp)
    cd_e = y_pre(cD, "-E", ftp, "Apoptosis", all_lines, itp=itp)
    cd_ae = y_pre(cD, "A/E", ftp, "Apoptosis", all_lines, itp=itp)

    r1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR1_RWD.csv")
    r2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR2_RWD.csv")
    r3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR3_RWD.csv")
    r4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR4_RWD.csv")
    ftp = 12
    cm = [r1, r2, r3, r4]
    m_ut = y_pre(cm, "UT", ftp, "Migration", all_lines)
    m_e = y_pre(cm, " E", ftp, "Migration", all_lines)
    m_ae = y_pre(cm, "A/E", ftp, "Migration", all_lines)

    m_ut.index = v_ut.index
    m_e.index = v_e.index
    m_ae.index = v_ae.index

    # Clustering Effect
    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    treatments = ['ut', 'e', 'ae']
    replicates = 6
    radius = np.linspace(1, 14.67, 1)
    folder = '48hrs'
    c = DataFrameRipleysK(folder, mutants, treatments, replicates, radius).reset_index().set_index("Mutant")
    c.columns = ["Treatment", "Island"]
    c_ut = c[c["Treatment"] == "ut"]
    c_ut = c_ut.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    c_ut.index = all_lines
    c_ut = c_ut.reset_index()
    c_ut["Treatment"] = "UT"

    c_e = c[c["Treatment"] == "e"]
    c_e = c_e.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    c_e.index = all_lines
    c_e = c_e.reset_index()
    c_e["Treatment"] = "E"

    c_ae = c[c["Treatment"] == "ae"]
    c_ae = c_ae.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    c_ae.index = all_lines
    c_ae = c_ae.reset_index()
    c_ae["Treatment"] = "A/E"

    # -------- PLOTS -------- #
    # PCA analysis of phenotypes
    y_ae = pd.concat([v_ae, cd_ae["Apoptosis"], m_ae["Migration"], c_ae["Island"]], axis=1)
    y_e = pd.concat([v_e, cd_e["Apoptosis"], m_e["Migration"], c_ae["Island"]], axis=1)
    y_ut = pd.concat([v_ut, cd_ut["Apoptosis"], m_ut["Migration"], c_ae["Island"]], axis=1)

    y_c = pd.concat([y_ut, y_e, y_ae])
    y_c.iloc[:, 2:] = StandardScaler().fit_transform(y_c.iloc[:, 2:])

    plotPCA(ax[:2], y_c, 3, ["Lines", "Treatment"], "Phenotype", hue_scores="Lines", style_scores="Treatment", hue_load="Phenotype", legendOut=True)

    # MODEL
    y = y_ae.drop("Treatment", axis=1).set_index("Lines")

    # -------- Cross-validation 1 -------- #
    # R2Y/Q2Y
    distance_method = "PAM250"
    ncl = 6
    SeqWeight = 0.5
    ncomp = 2

    MSC = MassSpecClustering(i, ncl, SeqWeight=SeqWeight, distance_method=distance_method, n_runs=1).fit(d, y)
    centers = MSC.transform(d)

    plsr = PLSRegression(n_components=ncomp, scale=False)
    plotR2YQ2Y(ax[2], plsr, centers, y, 1, 5)

    # Plot Measured vs Predicted
    plotActualVsPredicted(ax[3:7], plsr, centers, y, 1)

    # -------- Cross-validation 2 -------- #

    CoCl_plsr = Pipeline([("CoCl", MassSpecClustering(i, ncl, SeqWeight=SeqWeight, distance_method=distance_method)), ("plsr", PLSRegression(ncomp))])
    fit = CoCl_plsr.fit(d, y)
    centers = CoCl_plsr.named_steps.CoCl.transform(d)
    plotR2YQ2Y(ax[7], CoCl_plsr, d, y, cv=2, b=ncl + 1)
    plotActualVsPredicted(ax[9:13], CoCl_plsr, d, y, 2)
    plotScoresLoadings(ax[13:15], fit, centers, y, ncl, all_lines, 2)
    plotclusteraverages(ax[15], centers.T, all_lines)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotPCA(ax, d, n_components, scores_ind, loadings_ind, hue_scores=None, style_scores=None, hue_load=None, style_load=None, legendOut=False):
    """ Plot PCA scores and loadings. """
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.select_dtypes(include=["float64"]).values)
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, scores_ind, loadings_ind)
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    sns.scatterplot(x="PC1", y="PC2", data=dScor_, hue=hue_scores, style=style_scores, ax=ax[0], **{"linewidth": 0.5, "edgecolor": "k"})
    ax[0].set_title("PCA Scores", fontsize=11)
    ax[0].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[0].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    if legendOut:
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2)

    # Loadings
    g = sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue=hue_load, style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})
    ax[1].set_title("PCA Loadings", fontsize=11)
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    ax[1].get_legend().remove()
    for j, txt in enumerate(dLoad_[hue_load]):
        ax[1].annotate(txt, (dLoad_["PC1"][j] + 0.01, dLoad_["PC2"][j] + 0.01))


def plotGridSearch(ax, gs):
    """ Plot gridsearch results by ranking. """
    ax = sns.barplot(x="rank_test_score", y="mean_test_score", data=np.abs(gs.iloc[:20, :]), ax=ax, **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax.set_title("Hyperaparameter Search")
    ax.set_xticklabels(np.arange(1, 21))
    ax.set_ylabel("Mean Squared Error")


def plotR2YQ2Y(ax, model, X, Y, cv, b=3):
    """ Plot R2Y/Q2Y variance explained by each component. """
    Q2Y = R2Y_across_components(model, X, Y, cv, b, crossval=True)
    R2Y = R2Y_across_components(model, X, Y, cv, b)

    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align="center", label="Q2Y", color="darkblue")
    ax.bar(range_ - 0.15, R2Y, width=0.3, align="center", label="R2Y", color="black")
    ax.set_title("R2Y/Q2Y - Cross-validation strategy: " + str(cv), fontsize=12)
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Variance", fontsize=11)
    ax.legend(loc=0)


def plotActualVsPredicted(ax, plsr_model, X, Y, cv, y_pred="cross-validation"):
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
            ax[i].scatter(y, ypred)
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


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, cv, pcX=1, pcY=2, data="clusters", annotate=True):
    if cv == 1:
        X_scores, _ = model.transform(X, Y)
        PC1_xload, PC2_xload = model.x_loadings_[:, pcX - 1], model.x_loadings_[:, pcY - 1]
        PC1_yload, PC2_yload = model.y_loadings_[:, pcX - 1], model.y_loadings_[:, pcY - 1]

    if cv == 2:
        X_scores, _ = model.named_steps.plsr.transform(X, Y)
        PC1_xload, PC2_xload = model.named_steps.plsr.x_loadings_[:, pcX - 1], model.named_steps.plsr.x_loadings_[:, pcY - 1]
        PC1_yload, PC2_yload = model.named_steps.plsr.y_loadings_[:, pcX - 1], model.named_steps.plsr.y_loadings_[:, pcY - 1]

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


#     spacer = 0.1
#     ax[1].set_xlim([(-1 * max(np.abs(list(PC1_xload) + list(PC1_yload)))) - spacer, max(np.abs(list(PC1_xload) + list(PC1_yload))) + spacer])
#     ax[1].set_ylim([(-1 * max(np.abs(list(PC2_xload) + list(PC2_yload)))) - spacer, max(np.abs(list(PC2_xload) + list(PC2_yload))) + spacer])


def plotScoresLoadings_plotly(model, X, Y, cv, loc=False):
    """ Interactive PLSR plot. Note that this works best by pre-defining the dataframe's
    indices which will serve as labels for each dot in the plot. """
    if cv == 1:
        X_scores, _ = model.transform(X, Y)
        PC1_xload, PC2_xload = model.x_loadings_[:, 0], model.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.y_loadings_[:, 0], model.y_loadings_[:, 1]

    if cv == 2:
        X_scores, _ = model.named_steps.plsr.transform(X, Y)
        PC1_xload, PC2_xload = model.named_steps.plsr.x_loadings_[:, 0], model.named_steps.plsr.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.named_steps.plsr.y_loadings_[:, 0], model.named_steps.plsr.y_loadings_[:, 1]

    scores = pd.DataFrame()
    scores["PC1"] = X_scores[:, 0]
    scores["PC2"] = X_scores[:, 1]
    scores.index = X.index

    xloads = pd.DataFrame()
    xloads["PC1"] = PC1_xload
    xloads["PC2"] = PC2_xload
    xloads.index = X.columns

    yloads = pd.DataFrame()
    yloads["PC1"] = PC1_yload
    yloads["PC2"] = PC2_yload
    yloads.index = Y.columns

    if loc:
        print(xloads.loc[loc])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PLSR Scores", "PLSR Loadings"))
    fig.add_trace(
        go.Scatter(
            mode="markers+text",
            x=scores["PC1"],
            y=scores["PC2"],
            text=scores.index,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(color="blue", size=8, line=dict(color="black", width=1)),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=xloads["PC1"],
            y=xloads["PC2"],
            opacity=0.7,
            text=["Protein: " + xloads.index[i][0] + "  Pos: " + xloads.index[i][1] for i in range(len(xloads.index))],
            marker=dict(size=8, line=dict(color="black", width=1)),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=yloads["PC1"],
            y=yloads["PC2"],
            opacity=0.7,
            text=yloads.index,
            marker=dict(color=["green", "black", "blue", "cyan"], size=10, line=dict(color="black", width=1)),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        width=1000,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
    )
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)

    fig.show()
    return fig


def plotclusteraverages(ax, centers, treatments):

    colors_ = cm.rainbow(np.linspace(0, 1, centers.shape[0]))

    for i in range(centers.shape[0]):
        ax.plot(centers.iloc[i, :], marker="o", label="cluster " + str(i + 1), color=colors_[i])

    ax.set_xticks(np.arange(centers.shape[1]))
    ax.set_xticklabels(treatments, rotation=45)
    ax.set_ylabel("Normalized Signal", fontsize=12)
    ax.legend()


def plotKmeansPLSR_GridSearch(ax, X, Y):
    CVresults_max, CVresults_min, best_params = kmeansPLSR_tuning(X, Y)
    twoC = np.abs(CVresults_min.iloc[:2, 3])
    threeC = np.abs(CVresults_min.iloc[2:5, 3])
    fourC = np.abs(CVresults_min.iloc[5:9, 3])
    fiveC = np.abs(CVresults_min.iloc[9:14, 3])
    sixC = np.abs(CVresults_min.iloc[14:20, 3])

    width = 1
    groupgap = 1

    x1 = np.arange(len(twoC))
    x2 = np.arange(len(threeC)) + groupgap + len(twoC)
    x3 = np.arange(len(fourC)) + groupgap * 2 + len(twoC) + len(threeC)
    x4 = np.arange(len(fiveC)) + groupgap * 3 + len(twoC) + len(threeC) + len(fourC)
    x5 = np.arange(len(sixC)) + groupgap * 4 + len(twoC) + len(threeC) + len(fourC) + len(fiveC)

    ax.bar(x1, twoC, width, edgecolor="black", color="g")
    ax.bar(x2, threeC, width, edgecolor="black", color="g")
    ax.bar(x3, fourC, width, edgecolor="black", color="g")
    ax.bar(x4, fiveC, width, edgecolor="black", color="g")
    ax.bar(x5, sixC, width, edgecolor="black", color="g")

    comps = []
    for ii in range(2, 7):
        comps.append(list(np.arange(1, ii + 1)))
    flattened = [nr for cluster in comps for nr in cluster]

    ax.set_xticks(np.concatenate((x1, x2, x3, x4, x5)))
    ax.set_xticklabels(flattened, fontsize=10)
    ax.set_xlabel("Number of Components per Cluster")
    ax.set_ylabel("Mean-Squared Error (MSE)")


def plotClusters(X, cl_labels, nrows, ncols, xlabels, figsize=(15, 15)):
    """Boxplot of every cluster"""
    X["Cluster"] = cl_labels
    n = max(X["Cluster"])
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=True, figsize=figsize)
    for i in range(n):
        cl = X[X["Cluster"] == i + 1]
        m = pd.melt(cl, value_vars=list(cl.select_dtypes(include=["float"])), value_name="p-signal", id_vars=["Gene"], var_name="Lines")
        m["p-signal"] = m["p-signal"].astype("float64")
        sns.lineplot(x="Lines", y="p-signal", data=m, color="#658cbb", ax=ax[i // ncols][i % ncols], linewidth=2)
        ax[i // ncols][i % ncols].set_xticks(np.arange(len(xlabels)))
        ax[i // ncols][i % ncols].set_xticklabels(xlabels, rotation=45)
        ax[i // ncols][i % ncols].set_ylabel("$log_{10}$ p-signal")
        ax[i // ncols][i % ncols].xaxis.set_tick_params(bottom=True)
        ax[i // ncols][i % ncols].set_xlabel("")
        ax[i // ncols][i % ncols].legend(["cluster " + str(i + 1)])


def ArtificialMissingness(x, weights, nan_per, distance_method, ncl):
    """Incorporate different percentages of missing values and compute error between the actual
    versus cluster average value. Note that this works best with a complete subset of the CPTAC data set"""
    x.index = np.arange(x.shape[0])
    wlabels = ["Data", "Co-Clustering", "Sequence"]
    nan_indices = []
    errors = []
    missing = []
    prioritize = []
    n = x.iloc[:, 4:].shape[1]
    for per in nan_per:
        print(per)
        md = x.copy()
        m = int(n * per)
        for i in range(md.shape[0]):
            row_len = np.arange(4, md.shape[1])
            cols = random.sample(list(row_len), m)
            md.iloc[i, cols] = np.nan
            nan_indices.append((i, cols))
        for i, w in enumerate(weights):
            print(w)
            prioritize.append(wlabels[i])
            missing.append(per)
            errors.append(FitModelandComputeError(md, w, x, nan_indices, distance_method, ncl))

    X = pd.DataFrame()
    X["Prioritize"] = prioritize
    X["Missing%"] = missing
    X["Error"] = errors
    return X


def FitModelandComputeError(md, weight, x, nan_indices, distance_method, ncl):
    """Fit model and compute error during ArtificialMissingness"""
    i = md.select_dtypes(include=['object'])
    d = md.select_dtypes(include=['float64']).T
    model = MassSpecClustering(i, ncl, SeqWeight=weight, distance_method=distance_method, n_runs=1).fit(d, "NA")
    print(model.wins_)
    z = x.copy()
    z["Cluster"] = model.labels_
    centers = model.transform(d).T  # Clusters x observations
    errors = []
    for idx in nan_indices:
        v = z.iloc[idx[0], idx[1]]
        c = centers.iloc[z["Cluster"].iloc[idx[0]], np.array(idx[1]) - 4]
        errors.append(mean_squared_error(v, c))
    return np.mean(errors)


def WinsByWeight(i, d, weigths, distance_method):
    """Plot sequence, data, both, or mix score wins when fitting across a given set of weigths. """
    wins = []
    prioritize = []
    W = []
    for w in weigths:
        print(w)
        model = MassSpecClustering(i, ncl, SeqWeight=w, distance_method=distance_method, n_runs=1).fit(d, "NA")
        won = model.wins_
        W.append(w)
        wins.append(int(won.split("SeqWins: ")[1].split(" DataWins:")[0]))
        prioritize.append("Sequence")
        W.append(w)
        wins.append(int(won.split("DataWins: ")[1].split(" BothWin:")[0]))
        prioritize.append("Data")
        W.append(w)
        wins.append(int(won.split("BothWin: ")[1].split(" MixWin:")[0]))
        prioritize.append("Both")
        W.append(w)
        wins.append(int(won.split(" MixWin: ")[1]))
        prioritize.append("Mix")

    X = pd.DataFrame()
    X["Sequence_Weighting"] = W
    X["Prioritize"] = prioritize
    X["Wins"] = wins
    return X
