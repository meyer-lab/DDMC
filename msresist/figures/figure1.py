"""
This creates Figure 1.
"""
from .common import subplotLabel, getSetup
from ..sequence_analysis import FormatName, pYmotifs
from ..pre_processing import preprocessing, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.image as mpimg
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set(color_codes=True)


path = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.max_columns', 30)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 13), (4, 3))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in Cell Viability data
    r1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR1_Phase.csv")
    r2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR2_Phase.csv")
    r3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")

    itp = 24
    ftp = 72
    lines = ["PC9", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]

    # Read in Mass Spec data
    A = preprocessing(Axlmuts_ErlF154=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, FCtoUT=False, mc_row=True)
    A.columns = list(A.columns[:5]) + ["PC9", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    A = A[list(A.columns[:5]) + lines]

    # A: Cell Viability
    tr1 = ["-UT", '-E', '-A/E']
    tr2 = ["Untreated", 'Erlotinib', 'Erl + AF154']
    ylabel = "fold-change to t=" + str(itp) + "h"
    title = "Cell Viability - Erl + AF154 "
    c = ["white", "windows blue", "scarlet"]
    FC_timecourse(ax[0], r2, itp, ftp, lines, "A/E", title, ylabel, r2=r3, FC=True)
    barplot_UtErlAF154(ax[1], lines, r1, itp, ftp, tr1, tr2, "fold-change to t=0h", "Cell Viability", r2=r2, r3=r3, FC=True, colors=c)

    # blank out first two axis of the third column for reduced Viability-specific signaling ClusterMap
    hm_af154 = mpimg.imread('msresist/data/Signaling/CV_reducedHM_AF154.png')
    hm_erl = mpimg.imread('msresist/data/Signaling/CV_reducedHM_Erl.png')
    ax[2].imshow(hm_af154)
    ax[2].axis("off")
    ax[3].imshow(hm_erl)
    ax[3].axis("off")

    # Scores and Loadings MS data
    A = A.drop(["PC9"], axis=1)
    d = A.select_dtypes(include=['float64']).T
    plotpca_ScoresLoadings(ax[4:6], d, list(A["Abbv"]), list(A["Position"]))

    # Variability across overlapping peptides in MS replicates
#     X = preprocessing(Axlmuts_ErlF154=True, rawdata=True)
#     plotVarReplicates(ax[4:6], X)

    # Phosphorylation levels of selected peptides
    A = A[list(A.columns[:5]) + ["KI", "KO", "KD"] + lines[4:]]

    plot_AllSites(ax[6], A.copy(), "AXL", "AXL p-sites")

    RTKs = {"EGFR": "Y1197-p", "MET": "Y1003-p", "ERBB2": "Y877-p", "ERBB3": "Y1328-p", "EPHB3": "Y792-p"}
    plot_IdSites(ax[7], A.copy(), RTKs, "RTKs")

    adapters = {"GAB1": "Y627-p", "GAB2": "T265-p", "CRK": "Y251-p", "CRKL": "Y251-p", "SHC1": "S426-p"}
    plot_IdSites(ax[8], A.copy(), adapters, "Adapters")

    erks = {"MAPK3": "Y204-p;T202-p", "MAPK1": "Y187-p;T185-p", "MAPK7": "Y221-p"}
    erks_rn = {"MAPK3": "ERK1", "MAPK1": "ERK3", "MAPK7": "ERK5"}
    plot_IdSites(ax[9], A.copy(), erks, "ERK", erks_rn)

    jnks = {"MAPK9": "Y185-p", "MAPK10": "Y223-p"}
    jnks_rn = {"MAPK9": "JNK2", "MAPK10": "JNK3"}
    plot_IdSites(ax[10], A.copy(), jnks, "JNK", jnks_rn)

    p38s = {"MAPK12": "Y185-p", "MAPK13": "Y182-p", "MAPK14": "Y182-p"}
    p38s_rn = {"MAPK12": "P38G", "MAPK13": "P38D", "MAPK14": "P38A"}
    plot_IdSites(ax[11], A.copy(), p38s, "P38", p38s_rn)

    # Add subplot labels
    subplotLabel(ax)

    return f


def FC_timecourse(ax, r1, itp, ftp, lines, treatment, title, ylabel, r2=False, r3=False, FC=False):
    """ Fold-change time course of cell viability data. Initial and final time points must be specified. """
    if isinstance(r3, pd.core.frame.DataFrame):
        ds = [r1, r2, r3]
    if isinstance(r2, pd.core.frame.DataFrame):
        ds = [r1, r2]
    else:
        ds = [r1]

    c = []
    for i in range(len(ds)):
        # Compute fold-change
        if FC:
            for jj in range(1, ds[i].columns.size):
                ds[i].iloc[:, jj] /= ds[i][ds[i]["Elapsed"] == itp].iloc[0, jj]

        # Specify treatment
        r = ds[i].loc[:, ds[i].columns.str.contains(treatment)]
        r.columns = lines
        c.append(r)

    c = pd.concat(c, axis=1)
    c.insert(0, "Elapsed", r1.iloc[:, 0])
    c = c[c["Elapsed"] <= ftp]
    c = c[c["Elapsed"] >= itp]

    d = TransformTimeCourseMatrixForSeaborn(c, lines, itp, ylabel)

#     pal = sns.xkcd_palette(custom_colors)
    pal = sns.color_palette("Spectral", 10)
    sns.lineplot(x="Elapsed (h)", y=ylabel, hue="Lines", data=d, err_style="bars", ci='sd', ax=ax)

#     if treatment != "UT":
#         ax.legend().remove()

    ax.set_title(title)


def TransformTimeCourseMatrixForSeaborn(x, l, itp, ylabel):
    """ Preprocess data for seaborn. """
    y = pd.DataFrame()
    elapsed, lines, cv = [], [], []
    for idx, row in x.iterrows():
        df = pd.DataFrame(row).T
        elapsed.append(list(df["Elapsed"]) * (df.shape[1] - 1))
        lines.append(list(df.columns[1:]))
        cv.append(df.iloc[0, 1:].values)
    y["Elapsed (h)"] = [e for sl in elapsed for e in sl]
    y["Lines"] = [e for sl in lines for e in sl]
    y[ylabel] = [e for sl in cv for e in sl]
    return y


def timepoint_fc(d, itp, ftp):
    """ Calculate fold-change to specified time points, asserting no influnece of initial seeding. """
    dt0 = d[d["Elapsed"] == itp].iloc[0, 1:]
    dfc = d[d["Elapsed"] == ftp].iloc[0, 1:] / dt0

    # Assert that there's no significant influence of the initial seeding density
#     if itp < 12:
#         assert sp.stats.pearsonr(dt0, dfc)[1] > 0.05

    return pd.DataFrame(dfc).reset_index()


def FCendpoint(d, itp, ftp, t, l, ylabel, FC):
    """ Compute fold-change plus format for seaborn bar plot. """
    if FC:
        dfc = timepoint_fc(d, itp, ftp)
    else:
        dfc = pd.DataFrame(d[d["Elapsed"] == ftp].iloc[0, 1:]).reset_index()
        dfc.columns = ["index", 0]

    dfc["AXL mutants Y->F"] = l
    dfc["Treatment"] = t
    dfc = dfc[["index", "AXL mutants Y->F", "Treatment", 0]]
    dfc.columns = ["index", "AXL mutants Y->F", "Treatment", ylabel]
    return dfc.iloc[:, 1:]


def barplot_UtErlAF154(ax, lines, r1, itp, ftp, tr1, tr2, ylabel, title, r2=False, r3=False, FC=False, colors=colors):
    """ Cell viability bar plot at a specific end point across conditions, with error bars"""
    if isinstance(r3, pd.core.frame.DataFrame):
        ds = [r1, r2, r3]
    if isinstance(r2, pd.core.frame.DataFrame):
        ds = [r1, r2]
    else:
        ds = [r1]

    c = []
    for d in ds:
        for i, t in enumerate(tr1):
            x = pd.concat([d.iloc[:, 0], d.loc[:, d.columns.str.contains(t)]], axis=1)
            x = FCendpoint(x, itp, ftp, [tr2[i]] * 10, lines, ylabel, FC)
            c.append(x)

    c = pd.concat(c)
    pal = sns.xkcd_palette(colors)
    ax = sns.barplot(x="AXL mutants Y->F", y=ylabel, hue="Treatment", data=c, ci="sd", ax=ax, palette=pal, **{"linewidth": .5}, **{"edgecolor": "black"})

    ax.set_title(title)
    ax.set_xticklabels(lines, rotation=45)


def barplotFC_TvsUT(ax, r1, itp, ftp, l, tr1, tr2, title, r2=False, r3=False, FC=False, colors=colors):
    """ Bar plot of erl and erl + AF154 fold-change to untreated across cell lines. """
    if isinstance(r3, pd.core.frame.DataFrame):
        ds = [r1, r2, r3]
    if isinstance(r2, pd.core.frame.DataFrame):
        ds = [r1, r2]
    else:
        ds = [r1]

    c = []
    for d in ds:
        for j in range(1, len(tr1)):
            x = fc_TvsUT(d, itp, ftp, l, j, tr1, tr2[j], FC)
            c.append(x)

    c = pd.concat(c)
    pal = sns.xkcd_palette(colors)
    ax = sns.barplot(x="AXL mutants Y->F", y="fold-change to UT", hue="Treatment", data=c, ci="sd", ax=ax, palette=pal, **{"linewidth": .5}, **{"edgecolor": "black"})
    sns.set_context(rc={'patch.linewidth': 5})
    ax.set_title(title)
    ax.set_xticklabels(l, rotation=45)


def fc_TvsUT(d, itp, ftp, l, j, tr1, tr2, FC):
    """ Preprocess fold-change to untreated. """
    ut = pd.concat([d.iloc[:, 0], d.loc[:, d.columns.str.contains(tr1[0])]], axis=1)
    x = pd.concat([d.iloc[:, 0], d.loc[:, d.columns.str.contains(tr1[j])]], axis=1)

    if FC:
        ut = timepoint_fc(ut, itp, ftp).iloc[:, 1]
        x = timepoint_fc(x, itp, ftp).iloc[:, 1]

    else:
        ut = ut[ut["Elapsed"] == ftp].iloc[0, 1:].reset_index(drop=True)
        x = x[x["Elapsed"] == ftp].iloc[0, 1:].reset_index(drop=True)

    fc = pd.DataFrame(x.div(ut)).reset_index()

    fc["AXL mutants Y->F"] = l
    fc["Treatment"] = tr2
    fc = fc[["index", "AXL mutants Y->F", "Treatment", fc.columns[1]]]
    fc.columns = ["index", "AXL mutants Y->F", "Treatment", "fold-change to UT"]
    return fc


# Plot Separately since makefigure can't add it as a subplot
def plotClustergram(data, title, lim=False, robust=True, figsize=(10, 10)):
    """ Clustergram plot. """
    g = sns.clustermap(
        data,
        method="complete",
        cmap="bwr",
        robust=robust,
        vmax=lim,
        vmin=-lim,
        figsize=figsize)
    g.fig.suptitle(title, fontsize=17)
    ax = g.ax_heatmap
    ax.set_ylabel("")


def plotpca_explained(ax, data, ncomp):
    """ Cumulative variance explained for each principal component. """
    explained = PCA(n_components=ncomp).fit(data).explained_variance_ratio_
    acc_expl = []

    for i, exp in enumerate(explained):
        if i > 0:
            exp += acc_expl[i - 1]
            acc_expl.append(exp)
        else:
            acc_expl.append(exp)

    ax.bar(range(ncomp), acc_expl, edgecolor="black", linewidth=1)
    ax.set_ylabel("% Variance Explained")
    ax.set_xlabel("Components")
    ax.set_xticks(range(ncomp))
    ax.set_xticklabels([i + 1 for i in range(ncomp)])


def plotpca_ScoresLoadings(ax, data, pn, ps):
    fit = PCA(n_components=2).fit(data)
    PC1_scores, PC2_scores = fit.transform(data)[:, 0], fit.transform(data)[:, 1]
    PC1_loadings, PC2_loadings = fit.components_[0], fit.components_[1]

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores, linewidths=0.2)
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
    poi = [
        "AXL Y702-p",
        "AXL Y866-p",
        "AXL Y481-p",
        "RAB13 Y5-p",
        "RAB2B Y3-p",
        "CBLB Y889-p",
        "SOS1 Y1196-p",
        "GAB2 T265-p",
        "GAB1 Y406",
        "CRKL Y251-p",
        "PACSIN2 Y388-p",
        "SHC1 S426-p",
        "GSK3A Y279-p",
        "NME2 Y52-p",
        "CDK1 Y15-p",
        "MAPK3 T207-p",
        "TNS3 Y802-p",
        "GIT1 Y383-p",
        "KIRREL1 Y622-p",
        "BCAR1 Y234-p",
        "NEDD9 S182-p",
        "RIN1 Y681-p",
        "ATP8B1 Y1217",
        "MAPK3 Y204-p",
        "ATP1A1 Y55-p",
        "YES1 Y446",
        "EPHB3 Y792-p",
        "SLC35E1 Y403-p"]
    for i, name in enumerate(pn):
        p = name + " " + ps[i]
        if p in poi:
            ax[1].annotate(name, (PC1_loadings[i], PC2_loadings[i]))
    ax[1].scatter(PC1_loadings, PC2_loadings, c="darkred", linewidths=0.2, alpha=0.7)
    ax[1].set_title('PCA Loadings')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].axhline(y=0, color='0.25', linestyle='--')
    ax[1].axvline(x=0, color='0.25', linestyle='--')

    spacer = 0.05
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
    """ Plot variability of overlapping peptides across MS biological replicates. """
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
    ax[0].hist(DupsTable_drop.iloc[:, -1], bins=n_bins, edgecolor="black", linewidth=1)
    ax[0].set_ylabel("Number of peptides")
    ax[0].set_xlabel("Pearson Correlation Coefficients (n=" + str(DupsTable_drop.shape[0]) + ")")
    textstr = "$r2$ mean = " + str(np.round(DupsTable_drop.iloc[:, -1].mean(), 2))
    props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
    ax[0].text(.03, .96, textstr, transform=ax[0].transAxes, verticalalignment='top', bbox=props)

    ax[1].hist(Stds.mean(axis=1), bins=n_bins, edgecolor="black", linewidth=1)
    ax[1].set_ylabel("Number of peptides")
    ax[1].set_xlabel("Mean of Standard Deviations (n=" + str(Stds.shape[0]) + ")")
    textstr = "$σ$ mean = " + str(np.round(np.mean(Stds.mean(axis=1)), 2))
    props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
    ax[1].text(.75, .96, textstr, transform=ax[1].transAxes, verticalalignment='top', bbox=props)


def plot_AllSites(ax, x, prot, title):
    """ Plot all phosphopeptides for a given protein. """
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
            ax.plot(d.iloc[i, :], marker=".", label=positions, color=colors_[i])
        else:
            ax.plot(d.iloc[i, :], marker=".", label=positions[i], color=colors_[i])

    ax.legend(loc=0)
    ax.set_xticklabels(x.columns[4:], rotation=45)
    ax.set_ylabel("$Log2$ (p-site)")
    ax.set_title(title)


def plot_IdSites(ax, x, d, title, rn=False):
    """ Plot a set of specified p-sites. 'd' should be a dictionary werein every item is a protein-position pair. """
    x = x.set_index(["Abbv", "Position"])
    n = list(d.keys())
    p = list(d.values())
    colors_ = cm.rainbow(np.linspace(0, 1, len(n)))
    for i in range(len(n)):
        c = x.loc[n[i], p[i]]
        assert not (c is None), "Peptide not found."
        if rn:
            ax.plot(c[4:], marker=".", label=rn[n[i]], color=colors_[i])
        if not rn:
            ax.plot(c[4:], marker=".", label=n[i], color=colors_[i])

    ax.legend(loc=0)
    ax.set_xticklabels(c.index[3:], rotation=45)
    ax.set_ylabel("$Log2$ (p-site)")
    ax.set_title(title)


def selectpeptides(x, koi):
    l = []
    for n, p in koi.items():
        try:
            if isinstance(p, list):
                for site in p:
                    try:
                        l.append(x.loc[str(n), str(site)])
                    except BaseException:
                        continue

            if isinstance(p, str):
                l.append(x.loc[str(n), str(p)])
        except BaseException:
            continue
    ms = pd.DataFrame(l)
    ms = pd.concat(l, axis=1).T.reset_index()
    ms.columns = x.reset_index().columns
    return ms
