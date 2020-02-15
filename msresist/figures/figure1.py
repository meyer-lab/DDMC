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
    ax, f = getSetup((12, 10), (4, 3))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in data
    BR1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR1.csv").iloc[:, 1:]
    BR2 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR2.csv').iloc[:, 1:]
    BR3 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR3.csv').iloc[:, 1:]
    
    BR1_UT = pd.concat([BR1.iloc[:, 0], BR1.loc[:, BR1.columns.str.contains('UT')]], axis=1)
    BR1_E = pd.concat([BR1.iloc[:, 0], BR1.loc[:, BR1.columns.str.contains('-E')]], axis=1)
    BR1_AE = pd.concat([BR1.iloc[:, 0], BR1.loc[:, BR1.columns.str.contains('-A/E')]], axis=1)

    BR2_UT = pd.concat([BR2.iloc[:, 0], BR2.loc[:, BR2.columns.str.contains('UT')]], axis=1)
    BR2_E = pd.concat([BR2.iloc[:, 0], BR2.loc[:, BR2.columns.str.contains('-E')]], axis=1)
    BR2_AE = pd.concat([BR2.iloc[:, 0], BR2.loc[:, BR2.columns.str.contains('-A/E')]], axis=1)

    BR3_UT = pd.concat([BR2.iloc[:, 0], BR3.loc[:, BR3.columns.str.contains('UT')]], axis=1)
    BR3_E = pd.concat([BR2.iloc[:, 0], BR3.loc[:, BR3.columns.str.contains('-E')]], axis=1)
    BR3_AE = pd.concat([BR2.iloc[:, 0], BR3.loc[:, BR3.columns.str.contains('-A/E')]], axis=1)
    
    #A: Cell Viability EndPoint across mutants and treatments
    t = 120
    lines = ["PC9", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    BarPlot_UtErlAF154(ax[0], BR1_UT, BR2_UT, BR3_UT, BR1_E, BR2_E, BR3_E, BR1_AE, BR2_AE, BR3_AE, t, lines)

    E = preprocessing(Axlmuts_Erl=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, FCtoUT=False, mc_row=True).set_index(['Abbv', 'Sequence'])
    A = preprocessing(Axlmuts_ErlF154=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, FCtoUT=False, mc_row=True).set_index(['Abbv', 'Sequence'])
    d = A.select_dtypes(include=['float64']).T
    #B: blank out second axis for signaling ClusterMap
    ax[1].axis('off')
    
    #C&D: Scores and Loadings MS data
    plotpca_ScoresLoadings(ax[2:4], d)
    
    #E: Variability across overlapping peptides in MS replicates
    plotVarReplicates(ax, X)
    
    #F-: Phosphorylation levels of Selected peptides
    AXL(ax[4:7], E, A)

    EGFR(ax[7:9], E, A)

    OtherRTKs(ax[9], E, A)

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
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.set_title("Cell Viability-" + str(t) + "h " + title)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels(BR2.columns[1:], rotation=45)
    ax.bar(range_ - 0.20, BR1t135fc, width=width, align='center', label='BR1', color="black")
    ax.bar(range_ - 0.20, BR2fc, width=width, align='center', label='BR2', color="darkgreen")
    ax.bar(range_ + 0.20, BR3fc, width=width, align='center', label='BR3', color="darkblue")
    ax.legend()
    ax.set_ylabel("Fold-change to t=0h")


def BarPlot_UtErlAF154(ax, BR1_UT, BR2_UT, BR3_UT, BR1_E, BR2_E, BR3_E, BR1_AE, BR2_AE, BR3_AE, t, lines):
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

    spacer = 0.25
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
    spacer = 0.25
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


def AXL(ax, E, EA):
    E_axl759 = E.loc["AXL", "ENSEIyDYLRQ"][3:]
    E_axl866 = E.loc["AXL", "HPAGRyVLCPS"][3:]


    EA_axl759 = EA.loc["AXL", "ENSEIyDYLRQ"][3:]
    EA_axl866 = EA.loc["AXL", "HPAGRyVLCPS"][3:]
    EA_axl702 = EA.loc["AXL", "IYNGDyYRQGR"][3:]

    ax[0].plot(E_axl759, marker="o", label="erlotinib", color = "red")
    ax[0].plot(EA_axl759, marker="o", label="erl + AF154", color = "green")
    ax[0].set_title("AXL Y759-p", fontsize=15)
    ax[0].legend(loc=0)

    ax[1].plot(E_axl866, marker="o", color = "red", label="erlotinib")
    ax[1].plot(EA_axl866, marker="o", color = "green", label="erl + AF154")
    ax[1].set_title("AXL Y866-p", fontsize=15)
    ax[1].legend(loc=0)

    ax[2].plot(EA_axl702, marker="o", color="green", label="erl + AF154")
    ax[2].set_title("AXL Y702-p", fontsize=15)
    ax[2].legend(loc=0)

    ax[0].set_xticklabels(list(EA_axl702.index), rotation = 45)
    ax[1].set_xticklabels(list(EA_axl702.index), rotation = 45)
    ax[2].set_xticklabels(list(EA_axl702.index), rotation = 45)
    ax[0].set_ylabel("Normalized Signal", fontsize=15)


def EGFR(ax, E, EA):
    E_egfr1172 = E.loc["EGFR", "LDNPDyQQDFF"][3:]
    E_egfr1197 = E.loc["EGFR", "AENAEyLRVAP"][3:]

    EA_egfr1172 = EA.loc["EGFR", "LDNPDyQQDFF"][3:]
    EA_egfr1197 = EA.loc["EGFR", "AENAEyLRVAP"][3:]

    ax[0].plot(E_egfr1172, marker="o", label="erlotinib", color = "red")
    ax[0].plot(EA_egfr1172, marker="o", label="erl + AF154", color = "green")
    ax[0].set_title("EGFR Y1172-p", fontsize=15)
    ax[0].legend(loc=0)

    ax[1].plot(E_egfr1197, marker="o", color = "red", label="erlotinib")
    ax[1].plot(EA_egfr1197, marker="o", color = "green", label="erl + AF154")
    ax[1].set_title("EGFR Y1197-p", fontsize=15)
    ax[1].legend(loc=0)

    ax[0].set_ylabel("Normalized Signal", fontsize=15);
    ax[0].set_xticklabels(list(E_egfr1172.index), rotation = 45)
    ax[1].set_xticklabels(list(E_egfr1172.index), rotation = 45)


def OtherRTKs(ax, E, EA):
    E_met1003 = E.loc["MET", "NESVDyRATFP"][3:]
    EA_met1003 = EA.loc["MET", "NESVDyRATFP"][3:]

    E_erbb31328 = E.loc["ERBB3", "FDNPDyWHSRL"][3:]
    EA_erbb31328 = EA.loc["ERBB3", "FDNPDyWHSRL"][3:]

    EA_erbb2877 = EA.loc["ERBB2", "IDETEyHADGG"][3:]


    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20,5), sharex=True, sharey=True, squeeze=True)

    ax.plot(E_met1003, marker="o", color = "red", label="MET pY1003 Erlotinib")
    ax.plot(EA_met1003, marker="o", color = "green", label="MET pY1003 Erl + AF154")
    
    ax.plot(E_erbb31328, marker="o", color = "red", label="HER3 pY1328 erlotinib")
    ax.plot(EA_erbb31328, marker="o", color = "green", label="HER3 pY1328Erl + AF154")
    
    ax.plot(EA_erbb2877, marker="o", color="green", label="HER pY877 Erl + AF154")

    ax.legend(loc=0)
    ax.set_ylabel("Normalized Signal", fontsize=15)
    ax.set_xticklabels(list(E_met1003.index), rotation = 45)
    

def adapters(ax, E, EA):
    E_shb246 = E.loc["SHB", "TIADDySDPFD"][3:]
    E_stat3686 = E.loc["STAT3", "EAFGKyCRPES"][3:]
    E_shc1427 = E.loc["SHC1", "FDDPSyVNVQN"][3:]
    E_gab1659 = E.loc["GAB1", "DERVDyVVVDQ"][3:]
    E_gab2266 = E.loc["GAB2", "FRDSTyDLPRS"][3:]
    E_crk136 = E.loc["CRK", "QEEAEyVRALF"][3:]
    E_anxa2238 = E.loc["ANXA2", "KsYSPyDMLES"][3:]



    EA_shb246 = EA.loc["SHB", "TIADDySDPFD"][3:]
    EA_stat3686 = EA.loc["STAT3", "EAFGKyCRPES"][3:]
    EA_shc1427 = EA.loc["SHC1", "FDDPSyVNVQN"][3:]
    EA_gab1659 = EA.loc["GAB1", "DERVDyVVVDQ"][3:]
    EA_gab2266 = EA.loc["GAB2", "FRDSTyDLPRS"][3:]
    EA_crk136 = EA.loc["CRK", "QEEAEyVRALF"][3:]
    EA_anxa2238 = EA.loc["ANXA2", "KsYSPyDMLES"][3:]
    EA_dapp1 = EA.loc["DAPP1", "EEPSIyESVRV"][3:]

    ax[0, 0].plot(E_shb246, marker="x", color="gray", label="erlotinib")
    ax[0, 0].plot(EA_shb246, marker="o", color="darkorange", label="erl + AF154")
    ax[0, 0].set_title("SHB - Y246-p")
    ax[0, 0].legend(loc=0)
    ax[0, 0].set_ylabel("Normalized Signal")

    ax[0, 1].plot(E_stat3686, marker="x", color="gray", label="erlotinib")
    ax[0, 1].plot(EA_stat3686, marker="o", color="darkred", label="erl + AF154")
    ax[0, 1].set_title("SHB Y686-p")
    ax[0, 1].legend(loc=0)

    ax[0, 2].plot(E_shc1427, marker="x", color="gray", label="erlotinib")
    ax[0, 2].plot(EA_shc1427, marker="o", color="darkorange", label="erl + AF154")
    ax[0, 2].set_title("SHC1 Y427-p")
    ax[0, 2].legend(loc=0)

    ax[0, 3].plot(E_gab1659, marker="x", color="gray", label="erlotinib")
    ax[0, 3].plot(EA_gab1659, marker="o", color="darkblue", label="erl + AF154")
    ax[0, 3].set_title("GAB1 Y659-p")
    ax[0, 3].legend(loc=0)

    ax[1, 0].plot(E_gab2266, marker="x", color="gray", label="erlotinib")
    ax[1, 0].plot(EA_gab2266, marker="o", color="darkgreen", label="erl + AF154")
    ax[1, 0].set_title("GAB2 Y266-p")
    ax[1, 0].set_ylabel("Normalized Signal")
    ax[1, 0].legend(loc=0)

    ax[1, 1].plot(E_crk136, marker="x", color="gray", label="erlotinib")
    ax[1, 1].plot(EA_crk136, marker="o", color="darkcyan", label="erl + AF154")
    ax[1, 1].set_title("CRK Y136-p")
    ax[1, 1].legend(loc=0)

    ax[1, 2].plot(E_anxa2238, marker="x", color="gray", label="erlotinib")
    ax[1, 2].plot(EA_anxa2238, marker="o", color="brown", label="erl + AF154")
    ax[1, 2].set_title("ANXA2 Y238-p")
    ax[1, 2].legend(loc=0)

    ax[1, 3].plot(EA_dapp1, marker="o", color="red", label="erl + AF154")
    ax[1, 3].set_title("DAPP1 Y139-p")
    ax[1, 3].legend(loc=0)

    ax[1, 0].set_xticklabels(list(E_met1003.index), rotation = 45)
    ax[1, 1].set_xticklabels(list(E_met1003.index), rotation = 45)
    ax[1, 2].set_xticklabels(list(E_met1003.index), rotation = 45)
    ax[1, 3].set_xticklabels(list(E_met1003.index), rotation = 45)
    ax[0, 0].set_ylabel("Normalized Signal")
    ax[1, 0].set_ylabel("Normalized Signal")


def ERK(ax, E, EA):
    E_erk1s = E.loc["MAPK3", "GFLTEyVATRW"][3:]
    E_erk1d = E.loc["MAPK3", "GFLtEyVATRW"][3:]
    E_erk3s = E.loc["MAPK1", "GFLTEyVATRW"][3:]
    E_erk3d = E.loc["MAPK1", "GFLtEyVATRW"][3:]
    E_erk5s = E.loc["MAPK7", "YFMTEyVATRW"][3:]

    EA_erk1s = EA.loc["MAPK3", "GFLTEyVATRW"][3:]
    EA_erk1d = EA.loc["MAPK3", "GFLtEyVATRW"][3:]
    EA_erk3s = EA.loc["MAPK1", "GFLTEyVATRW"][3:]
    EA_erk3d = EA.loc["MAPK1", "GFLtEyVATRW"][3:]
    AE_erk5s = EA.loc["MAPK7", "YFMTEyVATRW"][3:]

    ax[0, 0].plot(E_erk1s, marker="x", color="gray", label="erlotinib")
    ax[0, 0].plot(EA_erk1s, marker="o", color="darkorange", label="erl + AF154")
    ax[0, 0].set_title("ERK1 - Y187-p")
    ax[0, 0].legend(loc=0)
    ax[0, 0].set_ylabel("normalized signal")
    ax[0, 0].set_xticklabels([])

    ax[0, 1].plot(E_erk1d, marker="x", color="gray")
    ax[0, 1].plot(EA_erk1d, marker="o", color="darkred")
    ax[0, 1].set_title("ERK1 - T185-p;Y187-p")
    ax[0, 1].set_xticklabels([])

    ax[0, 2].plot(E_erk3s, marker="x", color="gray")
    ax[0, 2].plot(EA_erk3s, marker="o", color="darkcyan")
    ax[0, 2].set_title("ERK3 - Y187-p")
    ax[0, 2].set_xticklabels(list(E_met1003.index), rotation = 45)


    ax[1, 0].plot(E_erk3d, marker="x", color="gray")
    ax[1, 0].plot(EA_erk3d, marker="o", color="darkblue")
    ax[1, 0].set_title("ERK3 - T185-p;Y187-p")

    ax[1, 1].plot(E_erk5s, marker="x", color="gray")
    ax[1, 1].plot(AE_erk5s, marker="o", color="darkgreen")
    ax[1, 1].set_title("ERK5 - Y221-p")

    ax[1, 2].remove()

    ax[1, 0].set_xticklabels(list(E_met1003.index), rotation = 45)
    ax[1, 1].set_xticklabels(list(E_met1003.index), rotation = 45)
    ax[0, 0].set_ylabel("Normalized Signal")
    ax[1, 0].set_ylabel("Normalized Signal")


def JNK(ax, E, EA):
    E_jnk2_185s = E.loc["MAPK9", "FMMTPyVVTRY"][3:]
    E_jnk2_223s = E.loc["MAPK10", "FMMTPyVVTRY"][3:]

    EA_jnk2_185s = EA.loc["MAPK9", "FMMTPyVVTRY"][3:]
    EA_jnk2_223s = EA.loc["MAPK10", "FMMTPyVVTRY"][3:]

    ax[0, 0].plot(E_jnk2_185s, marker="x", color="gray", label="erlotinib")
    ax[0, 0].plot(EA_jnk2_185s, marker="o", color="darkorange", label="erl + AF154")
    ax[0, 0].set_title("JNK2 Y185-p")
    ax[0, 0].legend(loc=0)
    ax[0, 0].set_ylabel("Normalized Signal")
    ax[0, 0].set_xticklabels([])

    ax[0, 1].plot(E_jnk2_223s, marker="x", color="gray")
    ax[0, 1].plot(EA_jnk2_223s, marker="o", color="darkred")
    ax[0, 1].set_title("JNK3 Y185-p")
    ax[0, 1].set_xticklabels([])

    ax[1, 0].set_xticklabels(list(E_jnk2_185s.index), rotation = 45)
    ax[1, 1].set_xticklabels(list(E_jnk2_185s.index), rotation = 45)
    ax[0, 0].set_ylabel("Normalized Signal")
    ax[1, 0].set_ylabel("Normalized Signal")


def p38(ax, E, EA):
    E_p38G_185 = E.loc["MAPK12", "SEMTGyVVTRW"][3:]
    E_p38D_182 = E.loc["MAPK13", "AEMTGyVVTRW"][3:]
    E_p38A_182 = E.loc["MAPK14", "DEMTGyVATRW"][3:]

    EA_p38G_185 = EA.loc["MAPK12", "SEMTGyVVTRW"][3:]
    EA_p38D_182 = EA.loc["MAPK13", "AEMTGyVVTRW"][3:]
    EA_p38A_182 = EA.loc["MAPK14", "DEMTGyVATRW"][3:]

    ax[0, 2].plot(E_p38G_185, marker="x", color="gray", label="erlotinib")
    ax[0, 2].plot(EA_p38G_185, marker="o", color="darkgreen", label="erl + AF154")
    ax[0, 2].set_title("P38G Y185-p")
    ax[0, 2].legend(loc=0)
    ax[0, 2].set_xticklabels(list(E_met1003.index), rotation = 45)


    ax[1, 0].plot(E_p38D_182, marker="x", color="gray")
    ax[1, 0].plot(EA_p38D_182, marker="o", color="darkblue")
    ax[1, 0].set_title("P38D Y182-p")
    ax[1, 0].set_ylabel("Normalized Signal")


    ax[1, 1].plot(E_p38A_182, marker="x", color="gray")
    ax[1, 1].plot(EA_p38A_182, marker="o", color="darkcyan")
    ax[1, 1].set_title("P38A Y182-p")