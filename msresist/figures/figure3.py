"""
This creates Figure 2.
"""
import os
import pandas as pd
import numpy as np
import scipy as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .common import subplotLabel, getSetup
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from msresist.clustering import MassSpecClustering
from msresist.parameter_tuning import MSclusPLSR_tuning, kmeansPLSR_tuning
from msresist.plsr import Q2Y_across_components, R2Y_across_components, Q2Y_across_comp_manual
from msresist.sequence_analysis import preprocess_seqs
from msresist.figures.figure1 import pca_dfs
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
from msresist.pre_processing import preprocessing, MergeDfbyMean, y_pre, FixColumnLabels
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 11), (3, 5))

    # blank out first axis for cartoon
#     ax[0].axis('off')

    # -------- Import and Preprocess Signaling Data -------- #
    X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)

    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    all_lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    mut_lines = all_lines[1:]
    g_lines = all_lines[2:]

    d.index = all_lines

    # -------- Cell Phenotypes -------- #
    # Cell Viability
    cv1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR1_Phase.csv")
    cv2 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR2_Phase.csv')
    cv3 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv')
    cv4 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv')

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

    # -------- PLOTS -------- #
    # PCA analysis of phenotypes
    y_ae = pd.concat([v_ae, cd_ae["Apoptosis"], m_ae["Migration"]], axis=1)
    y_e = pd.concat([v_e, cd_e["Apoptosis"], m_e["Migration"]], axis=1)
    y_ut = pd.concat([v_ut, cd_ut["Apoptosis"], m_ut["Migration"]], axis=1)

    y_fc = pd.concat([y_ae.iloc[:, :2], y_ae.iloc[:, 2:] / y_e.iloc[:, 2:]], axis=1)
    y_fc["Treatment"] = "A fold-change to E"

    PCA_scores(ax[:2], y_fc, 3)

    # MODEL
    y = y_fc.drop("Treatment", axis=1).set_index("Lines")

    # -------- Cross-validation 1 -------- #
    # R2Y/Q2Y
    distance_method = "PAM250"
    ncl = 10
    GMMweight = 10
    ncomp = 2

    MSC = MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method, n_runs=5).fit(d, y)
    centers = MSC.transform(d)

    plsr = PLSRegression(n_components=ncomp)
    plotR2YQ2Y(ax[2], plsr, centers, y, 1, 5)

    # Plot Measured vs Predicted
    plotActualVsPredicted(ax[3:6], plsr, centers, y, 1)

    # -------- Cross-validation 2 -------- #

    CoCl_plsr = Pipeline([('CoCl', MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method)), ('plsr', PLSRegression(ncomp))])
    fit = CoCl_plsr.fit(d, y)
    centers = CoCl_plsr.named_steps.CoCl.transform(d)
    plotR2YQ2Y(ax[6], CoCl_plsr, d, y, cv=2, b=ncl + 1)
    gs = pd.read_csv("msresist/data/Model/20200320-GridSearch_pam250_CVWC_wPC9.csv")
    gs[gs["#Components"] == 2].head(10)
    plotGridSearch(ax[7], gs)
    plotActualVsPredicted(ax[8:11], CoCl_plsr, d, y, 2)
    plotScoresLoadings(ax[11:13], fit, centers, y, ncl, all_lines, 2)
    plotclusteraverages(ax[13], centers.T, all_lines)

    # Add subplot labels
    subplotLabel(ax)

    return f


def PCA_scores(ax, d, n_components):
    """ Plot PCA scores. """
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.iloc[:, 2:].values)
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, ["Lines", "Treatment"], "Phenotype")
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    sns.scatterplot(x="PC1", y="PC2", data=dScor_, hue="Lines", ax=ax[0], s=80, **{'linewidth': .5, 'edgecolor': "k"})
    ax[0].set_title("PCA Scores", fontsize=11)
    ax[0].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[0].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2, fontsize=7)

    # Loadings
    sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="Phenotype", ax=ax[1], s=80, markers=["o", "X", "d"], **{'linewidth': .5, 'edgecolor': "k"})
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2, fontsize=7)
    ax[1].set_title("PCA Loadings", fontsize=11)
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)


def plotGridSearch(ax, gs):
    """ Plot gridsearch results by ranking. """
    ax = sns.barplot(x="Ranking", y="mean_test_scores", data=np.abs(gs.iloc[:20, :]), ax=ax, **{"linewidth": .5}, **{"edgecolor": "black"})
    ax.set_title("Hyperaparameter Search")
    ax.set_xticklabels(np.arange(1, 21))
    ax.set_ylabel("Mean Squared Error")


def plotR2YQ2Y(ax, model, X, Y, cv, b=3):
    """ Plot R2Y/Q2Y variance explained by each component. """
    Q2Y = Q2Y_across_comp_manual(model, X, Y, cv, b)
    R2Y = R2Y_across_components(model, X, Y, cv, b)

    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align='center', label='Q2Y', color="darkblue")
    ax.bar(range_ - 0.15, R2Y, width=0.3, align='center', label='R2Y', color="black")
    ax.set_title("R2Y/Q2Y - Cross-validation strategy: " + str(cv), fontsize=12)
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Variance", fontsize=11)
    ax.legend(loc=0)


def plotActualVsPredicted(ax, plsr_model, X, Y, cv, y_pred="cross-validation"):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    if y_pred == "cross-validation":
        #         Y_predictions = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
        cols = X.columns
        y_ = np.array(Y.copy().reset_index().drop("Lines", axis=1))
        X = np.array(X)
        Y_predictions = []
        if cv == 1:
            for train_index, test_index in LeaveOneOut().split(X, y_):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = y_[train_index], y_[test_index]
                Y_train = sp.stats.zscore(Y_train)
                plsr_model.fit(X_train, Y_train)
                Y_predict = list(plsr_model.predict(X_test).reshape(3,))
                Y_predictions.append(Y_predict)
        if cv == 2:
            for train_index, test_index in LeaveOneOut().split(X, y_):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = y_[train_index], y_[test_index]
                Y_train = sp.stats.zscore(Y_train)
                X_train = pd.DataFrame(X_train)
                X_train.columns = cols
                plsr_model.fit(pd.DataFrame(X_train), Y_train)
                Y_predict = list(plsr_model.predict(pd.DataFrame(X_test)).reshape(3,))
                Y_predictions.append(Y_predict)

        Y_predictions = np.array(Y_predictions)
        ylabel = "Predicted"
    if y_pred == "fit":
        Y_predictions = plsr_model.fit(X, Y).predict(X)
        ylabel = "Fit"
    for i, label in enumerate(Y.columns):
        y = Y.iloc[:, i]
        ypred = Y_predictions[:, i]
        ax[i].scatter(y, ypred)
        ax[i].plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
        ax[i].set_xlabel("Actual", fontsize=11)
        ax[i].set_ylabel(ylabel, fontsize=11)
        ax[i].set_title(label, fontsize=12)

        spacer = 1.1
        ax[i].set_xlim(min(list(y) + list(ypred)) * spacer, max(list(y) + list(ypred)) * spacer)
        ax[i].set_ylim(min(list(y) + list(ypred)) * spacer, max(list(y) + list(ypred)) * spacer)

        # Add correlation coefficient
        coeff, _ = sp.stats.pearsonr(ypred, y)
        textstr = "$r$ = " + str(np.round(coeff, 4))
        props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
        ax[i].text(0.80, 0.09, textstr, transform=ax[i].transAxes, fontsize=10, verticalalignment='top', bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, cv):
    if cv == 1:
        X_scores, _ = model.transform(X, Y)
        PC1_xload, PC2_xload = model.x_loadings_[:, 0], model.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.y_loadings_[:, 0], model.y_loadings_[:, 1]

    if cv == 2:
        X_scores, _ = model.named_steps.plsr.transform(X, Y)
        PC1_xload, PC2_xload = model.named_steps.plsr.x_loadings_[:, 0], model.named_steps.plsr.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.named_steps.plsr.y_loadings_[:, 0], model.named_steps.plsr.y_loadings_[:, 1]

    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]

    colors_ = cm.rainbow(np.linspace(0, 1, ncl))

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title('PLSR Model Scores', fontsize=12)
    ax[0].set_xlabel('Principal Component 1', fontsize=11)
    ax[0].set_ylabel('Principal Component 2', fontsize=11)
    ax[0].axhline(y=0, color='0.25', linestyle='--')
    ax[0].axvline(x=0, color='0.25', linestyle='--')

    spacer = 0.5
    ax[0].set_xlim([(-1 * max(np.abs(PC1_scores))) - spacer, max(np.abs(PC1_scores)) + spacer])
    ax[0].set_ylim([(-1 * max(np.abs(PC2_scores))) - spacer, max(np.abs(PC2_scores)) + spacer])

    # Loadings
    numbered = []
    list(map(lambda v: numbered.append(str(v + 1)), range(ncl)))
    for i, txt in enumerate(numbered):
        ax[1].annotate(txt, (PC1_xload[i], PC2_xload[i]))

    markers = ["x", "D", "*"]
    for i, label in enumerate(Y.columns):
        ax[1].annotate(label, (PC1_yload[i] + 0.05, PC2_yload[i] - 0.05))
        ax[1].scatter(PC1_yload[i], PC2_yload[i], color='black', marker=markers[i])

    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].set_title('PLSR Model Loadings (Averaged Clusters)', fontsize=12)
    ax[1].set_xlabel('Principal Component 1', fontsize=11)
    ax[1].set_ylabel('Principal Component 2', fontsize=11)
    ax[1].axhline(y=0, color='0.25', linestyle='--')
    ax[1].axvline(x=0, color='0.25', linestyle='--')

    spacer = 0.5
    ax[1].set_xlim([(-1 * max(np.abs(list(PC1_xload) + list(PC1_yload)))) - spacer, max(np.abs(list(PC1_xload) + list(PC1_yload))) + spacer])
    ax[1].set_ylim([(-1 * max(np.abs(list(PC2_xload) + list(PC2_yload)))) - spacer, max(np.abs(list(PC2_xload) + list(PC2_yload))) + spacer])


def plotScoresLoadings_plotly(X, labels, Y, ncomp, loc=False):
    """ Interactive PLSR plot. Note that this works best by pre-defining the dataframe's
    indices which will serve as labels for each dot in the plot. """

    plsr = PLSRegression(ncomp)
    X_scores, _ = plsr.fit_transform(X, Y)
    scores = pd.concat([pd.DataFrame(X_scores[:, 0]),
                        pd.DataFrame(X_scores[:, 1])], axis=1)
    scores.index = X.index
    scores.columns = ["PC1", "PC2"]

    xloads = pd.concat([pd.DataFrame(plsr.x_loadings_[:, 0]),
                        pd.DataFrame(plsr.x_loadings_[:, 1])], axis=1)
    yloads = pd.concat([pd.DataFrame(plsr.y_loadings_[:, 0]),
                        pd.DataFrame(plsr.y_loadings_[:, 1])], axis=1)

    xloads.index, yloads.index = X.columns, yloads.index.rename("Cell Viability")
    xloads.columns, yloads.columns = [["PC1", "PC2"]] * 2

    if loc:
        print(loadings.loc[loc])

    colors_ = ["black", "red", "blue", "lightgoldenrodyellow", "brown", "cyan", "orange", "gray"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PLSR Scores", "PLSR Loadings"))
    fig.add_trace(
        go.Scatter(
            mode='markers+text',
            x=scores["PC1"],
            y=scores["PC2"],
            text=scores.index,
            textposition="top center",
            textfont=dict(
                size=10,
                color="black"),
            marker=dict(
                color='blue',
                size=8,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=1)

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=xloads["PC1"],
            y=xloads["PC2"],
            opacity=0.7,
            text=["Protein: " + xloads.index[i][0] + "  Pos: " + xloads.index[i][1] for i in range(len(xloads.index))],
            marker=dict(
                color=[colors_[i] for i in labels],
                size=8,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=2)

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=yloads["PC1"],
            y=yloads["PC2"],
            opacity=0.7,
            text=yloads.index.name,
            marker=dict(
                color='green',
                size=10,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=2)

    fig.update_layout(
        height=500,
        width=1000,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False))
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)

    fig.show()


def plotclusteraverages(ax, centers, treatments):

    colors_ = cm.rainbow(np.linspace(0, 1, centers.shape[0]))

    for i in range(centers.shape[0]):
        ax.plot(centers.iloc[i, :], marker='o', label="cluster " + str(i + 1), color=colors_[i])

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

    ax.bar(x1, twoC, width, edgecolor='black', color="g")
    ax.bar(x2, threeC, width, edgecolor='black', color="g")
    ax.bar(x3, fourC, width, edgecolor='black', color="g")
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
