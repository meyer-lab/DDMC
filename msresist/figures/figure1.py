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
    ax, f = getSetup((16, 10), (3, 4))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in Cell Viability data
    BR1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR1.csv").iloc[:, 1:]
    BR2 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR2.csv').iloc[:, 1:]
    BR3 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR3.csv').iloc[:, 1:]

    t = 72
    lines = ["PC9", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    # Read in Mass Spec data
    E = preprocessing(Axlmuts_Erl=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, FCtoUT=False, mc_row=True).set_index(['Abbv', 'Sequence'])
    A = preprocessing(Axlmuts_ErlF154=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, FCtoUT=False, mc_row=True).set_index(['Abbv', 'Sequence'])
    E.columns = A.columns
    d = A.select_dtypes(include=['float64']).T
    
    #A: Cell Viability 
    BarPlot_UtErlAF154(ax[0], BR1, BR2, BR3, t, lines)

    #B: blank out second axis for signaling ClusterMap
    ax[1].axis('off')

    #C&D: Scores and Loadings MS data
    plotpca_ScoresLoadings(ax[2:4], d)

    #E: Variability across overlapping peptides in MS replicates
    X = preprocessing(Axlmuts_ErlF154=True, rawdata=True)
    plotVarReplicates(ax[4:6], X)

    #F-: Phosphorylation levels of selected peptides
    E = E.reset_index()
    A = A.reset_index()

    plotProteinSites(ax[6], A.copy(), "AXL", "AXL")
    plotProteinSitesEvsA(ax[7], E.copy(), A.copy(), "AXL", "AXL")

    plotProteinSites(ax[8], A.copy(), "EGFR", "EGFR")
    plotProteinSitesEvsA(ax[9], E.copy(), A.copy(), "EGFR", "EGFR")

    plotProteinSites(ax[10], A.copy(), "MAPK3", "ERK1")
    plotProteinSitesEvsA(ax[11], E.copy(), A.copy(), "MAPK3", "ERK1")

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


def FCendpoint(d, tp, t, l):
    dt0 = d[d["Elapsed"] == 0].iloc[0, 1:]
    dfc = d[d["Elapsed"] == tp].iloc[0, 1:] / dt0
    
    # Assert that there's no significant influence of the initial seeding density
    assert sp.stats.pearsonr(dt0, dfc)[1] > 0.05

    dfc = pd.DataFrame(dfc).reset_index()

    dfc["AXL mutants Y->F"] = l
    dfc["Treatment"] = t
    dfc = dfc[["index", "AXL mutants Y->F", "Treatment", 0]]
    dfc.columns = ["index", "AXL mutants Y->F", "Treatment", "Cell Viability (fold-change t=0)"]
    return dfc.iloc[:, 1:]


def plotReplicatesFoldChangeEndpoint(BR2, BR3, t, title):
    range_ = np.linspace(1, len(BR2.columns[1:]), len(BR2.columns[1:]))

    BR1t0 = BR1[BR1["Elapsed"] == 0].iloc[0, 1:]
    BR1t135fc = BR1[BR1["Elapsed"] == t].iloc[0, 1:] / BR1t0

    BR2t0 = BR2[BR2["Elapsed"] == 0].iloc[0, 1:]
    BR2fc = BR2[BR2["Elapsed"] == t].iloc[0, 1:] / BR2t0

    BR3t0 = BR3[BR3["Elapsed"] == 0].iloc[0, 1:]
    BR3fc = BR3[BR3["Elapsed"] == t].iloc[0, 1:] / BR3t0

    assert sp.stats.pearsonr(BR1t0, BR1t135fc)[1] > 0.05, (BR1t0, BR1t135fc)
    assert sp.stats.pearsonr(BR2t0, BR2fc)[1] > 0.05, (BR2t0, BR2fc)
    assert sp.stats.pearsonr(BR3t0, BR3fc)[1] > 0.05, (BR3t0, BR3fc)

    width=0.4
    ax.set_title("Cell Viability-" + str(t) + "h " + title)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels(BR2.columns[1:], rotation=45)
    ax.bar(range_ - 0.20, BR1t135fc, width=width, align='center', label='BR1', color="black")
    ax.bar(range_ - 0.20, BR2fc, width=width, align='center', label='BR2', color="darkgreen")
    ax.bar(range_ + 0.20, BR3fc, width=width, align='center', label='BR3', color="darkblue")
    ax.legend()
    ax.set_ylabel("Fold-change to t=0h")


def BarPlot_UtErlAF154(ax, BR1, BR2, BR3, t, lines):
    BR1_UT = pd.concat([BR1.iloc[:, 0], BR1.loc[:, BR1.columns.str.contains('UT')]], axis=1)
    BR1_E = pd.concat([BR1.iloc[:, 0], BR1.loc[:, BR1.columns.str.contains('-E')]], axis=1)
    BR1_AE = pd.concat([BR1.iloc[:, 0], BR1.loc[:, BR1.columns.str.contains('-A/E')]], axis=1)

    BR2_UT = pd.concat([BR2.iloc[:, 0], BR2.loc[:, BR2.columns.str.contains('UT')]], axis=1)
    BR2_E = pd.concat([BR2.iloc[:, 0], BR2.loc[:, BR2.columns.str.contains('-E')]], axis=1)
    BR2_AE = pd.concat([BR2.iloc[:, 0], BR2.loc[:, BR2.columns.str.contains('-A/E')]], axis=1)

    BR3_UT = pd.concat([BR2.iloc[:, 0], BR3.loc[:, BR3.columns.str.contains('UT')]], axis=1)
    BR3_E = pd.concat([BR2.iloc[:, 0], BR3.loc[:, BR3.columns.str.contains('-E')]], axis=1)
    BR3_AE = pd.concat([BR2.iloc[:, 0], BR3.loc[:, BR3.columns.str.contains('-A/E')]], axis=1)
    
    br1_ut = FCendpoint(BR1_UT, t, ["UT"]*10, lines)
    br2_ut = FCendpoint(BR2_UT, t, ["UT"]*10, lines)
    br3_ut = FCendpoint(BR3_UT, t, ["UT"]*10, lines)
    br1_e = FCendpoint(BR1_E, t, ["Erlotinib"]*10, lines)
    br2_e = FCendpoint(BR2_E, t, ["Erlotinib"]*10, lines)
    br3_e = FCendpoint(BR3_E, t, ["Erlotinib"]*10, lines)
    br1_ae = FCendpoint(BR1_AE, t, ["Erl + AF154"]*10, lines)
    br2_ae = FCendpoint(BR2_AE, t, ["Erl + AF154"]*10, lines)
    br3_ae = FCendpoint(BR3_AE, t, ["Erl + AF154"]*10, lines)
    c = pd.concat([br2_ut, br3_ut, br2_e, br3_e, br2_ae, br3_ae])

    ax = sns.barplot(x="AXL mutants Y->F", y="Cell Viability (fold-change t=0)", hue="Treatment", data=c, ci="sd")
    ax.set_title("t=" + str(t) + "h")
    ax.set_xticklabels(lines, rotation=45)


# Plot Separately
def plotClustergram(data, title, lim=False, robust=True):
    g = sns.clustermap(
        data,
        method="complete",
        cmap="bwr",
        robust=robust,
        vmax=lim,
        vmin=-lim)
    g.fig.suptitle(title, fontsize=17)
    ax = g.ax_heatmap
    ax.set_ylabel("")


def plotpca_explained(ax, data, ncomp):
    explained = PCA(n_components=ncomp).fit(data).explained_variance_ratio_
    acc_expl = []

    for i, exp in enumerate(explained):
        if i > 0:
            exp += acc_expl[i - 1]
            acc_expl.append(exp)
        else:
            acc_expl.append(exp)

    ax.bar(range(ncomp), acc_expl)
    ax.set_ylabel("% Variance Explained")
    ax.set_xlabel("Components")
    ax.set_xticks(range(ncomp))
    ax.set_xticklabels([i + 1 for i in range(ncomp)])


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
    ax[0].set_xlim([(-1 * max(np.abs(PC1_scores))) - spacer, max(np.abs(PC1_scores)) + spacer])
    ax[0].set_ylim([(-1 * max(np.abs(PC2_scores))) - spacer, max(np.abs(PC2_scores)) + spacer])

    # Loadings
    for i, txt in enumerate(list(data.columns)):
        ax[1].annotate(txt, (PC1_loadings[i], PC2_loadings[i]))
    ax[1].scatter(PC1_loadings, PC2_loadings, c=np.arange(PC1_loadings.size), cmap=colors.ListedColormap(colors_))
    ax[1].set_title('PCA Loadings')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].axhline(y=0, color='0.25', linestyle='--')
    ax[1].axvline(x=0, color='0.25', linestyle='--')
    spacer = 0.5
    ax[1].set_xlim([(-1 * max(np.abs(PC1_loadings)) - spacer), (max(np.abs(PC1_loadings)) + spacer)])
    ax[1].set_ylim([(-1 * max(np.abs(PC2_loadings)) - spacer), (max(np.abs(PC2_loadings)) + spacer)])


def plotpca_ScoresLoadings_plotly(data, title, loc=False):
    """ Interactive PCA plot. Note that this works best by pre-defining the dataframe's
    indices which will serve as labels for each dot in the plot. """
    fit = PCA(n_components=2).fit(data)

    scores = pd.concat([pd.DataFrame(fit.transform(data)[:, 0]), pd.DataFrame(fit.transform(data)[:, 1])], axis=1)
    scores.index = data.index
    scores.columns = ["PC1", "PC2"]

    loadings = pd.concat([pd.DataFrame(fit.components_[0]), pd.DataFrame(fit.components_[1])], axis=1)
    loadings.index = data.columns
    loadings.columns = ["PC1", "PC2"]

    if loc:
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
            text=["Protein: " + loadings.index[i][0] + "  Pos: " + loadings.index[i][1] for i in range(len(loadings.index))],
            marker=dict(
                color='crimson',
                size=8,
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
        yaxis2=dict(showgrid=False),
        title_text=title),
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)

    fig.show()


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


def plotProteinSites(ax, x, prot, title):
    "Plot all phosphopeptides for a given protein"
    x = x.set_index(["Abbv"])
    peptides = pd.DataFrame(x.loc[prot])
    assert peptides.shape[0] > 0
    if peptides.shape[1] == 1:
        peptides = peptides.T
        d = peptides.iloc[:, 4:]
    else:
        d = peptides.select_dtypes(include=['float64'])

    positions = x.loc[prot]["Position"]
    
    colors_ = cm.rainbow(np.linspace(0, 1, peptides.shape[0]))
    for i in range(peptides.shape[0]):
        if peptides.shape[0] == 1:
            ax.plot(d.iloc[i, :], marker="o", label=positions, color = colors_[i])
        else:
            ax.plot(d.iloc[i, :], marker="o", label=positions[i], color = colors_[i])
    
    ax.legend(loc=0)
    ax.set_xticklabels(x.columns[4:], rotation=45)
    ax.set_ylabel("Normalized Signal", fontsize=10)
    ax.set_title(title)


def plotProteinSitesEvsA(ax, E, A, prot, title):
    "Plot fold change between AF154 vs Erlotinib only"
    E.insert(5, "Treatment", ["erlotinib"]*E.shape[0])
    A.insert(5, "Treatment", ["AF154"]*A.shape[0])
    c = pd.concat([E, A])
    x = c.set_index(["Abbv"])

    peptides = pd.DataFrame(x.loc[prot])
    assert peptides.shape[0] > 0
    d = peptides.groupby(["Position", "Treatment"]).mean().reset_index().set_index("Position")

    fd, positions = [], []
    for pos in list(set(d.index)):
        x = d.loc[pos]
        if x.shape[0] == 2:
            positions.append(pos)
            fd.append(x[x["Treatment"]=="AF154"].iloc[0, 1:].div(x[x["Treatment"]=="erlotinib"].iloc[0, 1:]))

    colors_ = cm.rainbow(np.linspace(0, 1, len(positions)))
    for j in range(len(positions)):
        ax.plot(fd[j], marker="o", label=positions[j], color = colors_[j])

    ax.legend(loc=0)
    ax.set_ylabel("Fold-change AF154 vs Erl Only", fontsize=10)
    ax.set_xticklabels(E.columns[6:], rotation=45)
    ax.set_title(title)
