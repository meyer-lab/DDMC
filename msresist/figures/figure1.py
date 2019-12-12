"""
This creates Figure 1.
"""
from .common import subplotLabel, getSetup
from ..sequence_analysis import FormatName, pYmotifs
from ..pre_processing import preprocessing, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, MergeDfbyMean
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set(color_codes=True)


path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (3, 3))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in data
    Y_cv1 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw3.csv')).iloc[:30, :11]
    Y_cv2 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw4.csv')).iloc[:29, :11]

    # Assert that there's no significant influence of the initial seeding density
    Y_cvE3_0 = Y_cv1[Y_cv1["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:] / Y_cvE3_0

    Y_cvE4_0 = Y_cv2[Y_cv2["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE4 = Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:] / Y_cvE4_0

    assert sp.stats.pearsonr(Y_cvE3_0, Y_fcE3)[1] > 0.05
    assert sp.stats.pearsonr(Y_cvE4_0, Y_fcE4)[1] > 0.05

    # Normalize to t=0
    for ii in range(1, Y_cv2.columns.size):
        Y_cv1.iloc[:, ii] /= Y_cv1.iloc[0, ii]
        Y_cv2.iloc[:, ii] /= Y_cv2.iloc[0, ii]

    plotTimeCourse(ax[0:2], Y_cv1, Y_cv2)

    plotAveragedEndpoint(ax[2], Y_cv1, Y_cv2)

    plotRTKs(ax[3:7])

    X = preprocessing(AXLwt=True, rawdata=True)
    plotVarReplicates(ax[7:9], X)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotTimeCourse(ax, Y_cv1, Y_cv2):
    """ Plots the Incucyte timecourse. """
    ax[0].set_title("Experiment 3")
    ax[0].plot(Y_cv1.iloc[:, 0], Y_cv1.iloc[:, 1:])
    ax[0].legend(Y_cv1.columns[1:])
    ax[0].set_ylabel("Fold-change to t=0h")
    ax[0].set_xlabel("Time (hours)")
    ax[1].set_title("Experiment 4")
    ax[1].plot(Y_cv2.iloc[:, 0], Y_cv2.iloc[:, 1:])
    ax[1].set_ylabel("Fold-change to t=0h")
    ax[1].set_xlabel("Time (hours)")


def plotReplicatesEndpoint(ax, Y_cv1, Y_cv2):
    range_ = np.linspace(1, 10, 10)

    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:]
    Y_fcE4 = Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:]

    ax.set_title("Cell Viability - 72h")
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels(Y_cv1.columns[1:])
    ax.bar(range_ + 0.15, Y_fcE3, width=0.3, align='center', label='Exp3', color="black")
    ax.bar(range_ - 0.15, Y_fcE4, width=0.3, align='center', label='Exp4', color="darkred")
    ax.legend()
    ax.set_ylabel("% Confluency")


def plotReplicatesFoldChangeEndpoint(ax, Y_cv1, Y_cv2):
    range_ = np.linspace(1, 10, 10)

    Y_cvE3_0 = Y_cv1[Y_cv1["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:] / Y_cvE3_0

    Y_cvE4_0 = Y_cv2[Y_cv2["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE4 = Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:] / Y_cvE4_0

    ax.set_title("Cell Viability - 72h")
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels(Y_cv1.columns[1:])
    ax.bar(range_ + 0.15, Y_fcE3, width=0.3, align='center', label='Exp3', color="black")
    ax.bar(range_ - 0.15, Y_fcE4, width=0.3, align='center', label='Exp4', color="darkgreen")
    ax.legend()
    ax.set_ylabel("Fold-change 72h vs 0h")


def plotAveragedEndpoint(ax, Y_cv1, Y_cv2):
    range_ = np.linspace(1, 10, 10)

    Y_cv = MergeDfbyMean(pd.concat([Y_cv1, Y_cv2], axis=0), Y_cv1.columns, "Elapsed")
    Y_cv = Y_cv.reset_index()[Y_cv1.columns]
    Y_cv = Y_cv[Y_cv["Elapsed"] == 72].iloc[0, 1:]

    ax.set_title("Cell Viability - 72h")
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels(Y_cv1.columns[1:])
    ax.bar(range_, Y_cv, width=0.5, align='center', color="black")
    ax.set_ylabel("% Confluency")


def plotRTKs(ax):
    ABC = preprocessing(AXLwt=True)
    header = ABC.columns

    EGFR = ABC[ABC["Sequence"].str.contains("SHQISLDNPDyQQDFFP")].mean()
    IGFR = ABC[ABC["Sequence"].str.contains("IYETDYyR")].iloc[:, 3:13].mean()
    MET = ABC[ABC["Sequence"].str.contains("MYDkEyYSVHNk")].iloc[:, 3:13].mean()
    AXL = ABC[ABC["Sequence"].str.contains("YNGDyYR")].iloc[:, 3:13].mean()

    ax[0].set_title("EGFR: pY1172", fontsize=13)
    ax[0].plot(EGFR)
    ax[0].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')
    ax[1].set_title("IGFR: pY1190", fontsize=13)
    ax[1].plot(IGFR)
    ax[1].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')
    ax[2].set_title("MET: pY1234", fontsize=13)
    ax[2].plot(MET)
    ax[2].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')
    ax[3].set_title("AXL: pY702", fontsize=13)
    ax[3].plot(AXL)
    ax[3].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')


def plotVarReplicates(ax, ABC):
    ABC = pYmotifs(ABC, list(ABC.iloc[:, 0]))
    NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    # Correlation of Duplicates, optionally filtering first
    DupsTable = BuildMatrix(CorrCoefPeptides, ABC)
    # DupsTable = CorrCoefFilter(DupsTable)
    DupsTable_drop = DupsTable.drop_duplicates(["Protein", "Sequence"])
    assert(DupsTable.shape[0] / 2 == DupsTable_drop.shape[0])

    # Stdev of Triplicates, optionally filtering first
    StdPeptides = BuildMatrix(StdPeptides, ABC)
    TripsTable = TripsMeanAndStd(StdPeptides, list(ABC.columns[:3]))
    Stds = TripsTable.iloc[:, TripsTable.columns.get_level_values(1) == 'std']
    # Xidx = np.all(Stds.values <= 0.4, axis=1)
    # Stds = Stds.iloc[Xidx, :]

    n_bins = 10
    ax[0].hist(DupsTable_drop.iloc[:, -1], bins=n_bins)
    ax[0].set_ylabel("Number of peptides", fontsize=12)
    ax[0].set_xlabel("Pearson Correlation Coefficients N= " + str(DupsTable_drop.shape[0]), fontsize=12)
    textstr = "$r2$ mean = " + str(np.round(DupsTable_drop.iloc[:, -1].mean(), 2))
    props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
    ax[0].text(.03, .96, textstr, transform=ax[0].transAxes, fontsize=12, verticalalignment='top', bbox=props)

    ax[1].hist(Stds.mean(axis=1), bins=n_bins)
    ax[1].set_ylabel("Number of peptides", fontsize=12)
    ax[1].set_xlabel("Mean of Standard Deviations N= " + str(Stds.shape[0]), fontsize=12)
    textstr = "$σ$ mean = " + str(np.round(np.mean(Stds.mean(axis=1)), 2))
    props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
    ax[1].text(.8, .96, textstr, transform=ax[1].transAxes, fontsize=12, verticalalignment='top', bbox=props)


def plotClustergram(data, lim, title):
    g = sns.clustermap(
        data, 
        method="complete", 
        cmap="bwr", 
        robust=True,
        vmax=lim, 
        vmin=-lim)
    g.fig.suptitle(title, fontsize=17)


#     p = g.dendrogram_row.reordered_ind
#     corr = ABC_mc.iloc[p, 2:].T.corr(method='pearson')
#     Correlation heatmap was really just for exploration. Not including here.
#     ax[1] = sns.heatmap(corr,  vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True, ax=ax[1])
#     ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=80, horizontalalignment='right')
#     ax[1].set_title("Correlation Heatmap");


def plotpca_explained(ax, data, ncomp):
    explained = PCA(n_components=ncomp).fit(data).explained_variance_ratio_
    acc_expl = []

    for i, exp in enumerate(explained):
        if i > 0:
            exp+=acc_expl[i-1]
            acc_expl.append(exp)
        else:
            acc_expl.append(exp)

    ax.bar(range(ncomp), acc_expl)
    ax.set_ylabel("% Variance Explained")
    ax.set_xlabel("Components")
    ax.set_xticks(range(ncomp))
    ax.set_xticklabels([i+1 for i in range(ncomp)])


def plotpca_ScoresLoadings(ax, data):
    fit = PCA(n_components=2).fit(data)
    PC1_scores, PC2_scores = fit.transform(data)[:, 0], fit.transform(data)[:, 1]
    PC1_loadings, PC2_loadings = fit.components_[0], fit.components_[1]
    
    colors_ = cm.rainbow(np.linspace(0, 1, PC1_scores.size))

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(list(data.index)):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title('PCA Scores')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].axhline(y=0, color='0.25', linestyle='--')
    ax[0].axvline(x=0, color='0.25', linestyle='--')

    spacer = 0.5
    ax[0].set_xlim([(-1 * max(PC1_scores)) - spacer, max(PC1_scores) + spacer])
    ax[0].set_ylim([(-1 * max(PC2_scores)) - spacer, max(PC2_scores) + spacer])

    # Loadings
    for i, txt in enumerate(list(data.columns)):
        ax[1].annotate(txt, (PC1_loadings[i], PC2_loadings[i]))
    ax[1].scatter(PC1_loadings, PC2_loadings, c=np.arange(PC1_loadings.size), cmap=colors.ListedColormap(colors_))
    ax[1].set_title('PCA Loadings')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].axhline(y=0, color='0.25', linestyle='--')
    ax[1].axvline(x=0, color='0.25', linestyle='--')
    spacer = 0.04
    ax[1].set_xlim([(-1 * max(PC1_loadings) - spacer), (max(PC1_loadings) + spacer)])
    ax[1].set_ylim([(-1 * max(PC2_loadings) - spacer), (max(PC2_loadings) + spacer)]);


def plotpca_ScoresLoadings_plotly(data, title, loc=False):
    fit = PCA(n_components=2).fit(data)

    scores = pd.concat([pd.DataFrame(fit.transform(data)[:, 0]), pd.DataFrame(fit.transform(data)[:, 1])], axis=1)
    scores.index = data.index
    scores.columns = ["PC1", "PC2"]

    loadings = pd.concat([pd.DataFrame(fit.components_[0]), pd.DataFrame(fit.components_[1])], axis=1)
    loadings.index = data.columns
    loadings.columns = ["PC1", "PC2"]
    if loc:
        loadings = loadings.set_index(data.columns)
        print(loadings.loc[loc])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA Scores", "PCA Loadings"))
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
            x=loadings["PC1"],
            y=loadings["PC2"],
            opacity=0.7,
            text=loadings.index,
            marker=dict(
                color='crimson',
                size=8,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=2)
    
    fig.update_layout(
        height=650, 
        width=1250, 
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
        title_text=title),
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)

    fig.show()
