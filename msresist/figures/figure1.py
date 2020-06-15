"""
This creates Figure 1.
"""
from .common import subplotLabel, getSetup
from ..motifs import pYmotifs
from ..pre_processing import preprocessing, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, FixColumnLabels
from sklearn.decomposition import PCA
import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
sns.set(color_codes=True)


path = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.max_columns", 30)


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
    r4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR4_Phase.csv")

    itp = 24
    ftp = 72
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]

    # Read in Mass Spec data
    A = preprocessing(Axlmuts_ErlAF154=True, Vfilter=False, FCfilter=False, log2T=True, FCtoUT=False, mc_row=True)
    A.columns = list(A.columns[:5]) + ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    A = A[list(A.columns[:5]) + lines]

    # A: Cell Viability
    ds = [r1, r2, r3, r4]
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]
    ylabel = "fold-change to t=" + str(itp) + "h"
    title = "Cell Viability - Erl + AF154 "
    c = ["white", "windows blue", "scarlet"]
    IndividualTimeCourses(ds, ftp, lines, tr1, tr2, ylabel, TimePointFC=itp, TreatmentFC=False, plot="WT", ax_=ax[0])
    barplot_UtErlAF154(ax[1], lines, ds, ftp, tr1, tr2, "fold-change to t=0h", "Cell Viability", TimePointFC=itp, colors=c)

    # blank out first two axis of the third column for reduced Viability-specific signaling ClusterMap
    hm_af154 = mpimg.imread("msresist/data/MS/AXL/CV_reducedHM_AF154.png")
    hm_erl = mpimg.imread("msresist/data/MS/AXL/CV_reducedHM_Erl.png")
    ax[2].imshow(hm_af154)
    ax[2].axis("off")
    ax[3].imshow(hm_erl)
    ax[3].axis("off")

    # Scores and Loadings MS data
    A = A.drop(["WT"], axis=1)
    d = A.select_dtypes(include=["float64"]).T
    plotpca_ScoresLoadings(ax[4:6], d, list(A["Gene"]), list(A["Position"]))

    # Variability across overlapping peptides in MS replicates
    #     X = preprocessing(Axlmuts_ErlF154=True, rawdata=True)
    #     plotVarReplicates(ax[4:6], X)

    # Phosphorylation levels of selected peptides
    A = A[list(A.columns[:5]) + ["KI", "KO", "KD"] + lines[4:]]

    plot_AllSites(ax[6], A.copy(), "AXL", "AXL p-sites")

    RTKs = {"EGFR": "Y1197-p", "MET": "Y1003-p", "ERBB2": "Y877-p", "ERBB3": "Y1328-p", "EPHB3": "T791-p"}
    plot_IdSites(ax[7], A.copy(), RTKs, "RTKs")

    adapters = {"GAB1": "Y659-p", "GAB2": "T265-p", "CRK": "Y136-p", "CRKL": "Y251-p", "SHC1": "S426-p"}
    plot_IdSites(ax[8], A.copy(), adapters, "Adapters")

    erks = {"MAPK3": "Y204-p;T202-p", "MAPK1": "Y187-p;T185-p", "MAPK7": "Y221-p"}
    erks_rn = {"MAPK3": "ERK1", "MAPK1": "ERK2", "MAPK7": "ERK5"}

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


def IndividualTimeCourses(
    ds, ftp, lines, t1, t2, ylabel, TimePointFC=False, TreatmentFC=False, savefig=False, plot="Full", ax_=False, figsize=(20, 10)
):
    """ Plot time course data of each cell line across treatments individually. """
    ds = FixColumnLabels(ds)
    c = []
    for d in ds:
        if TimePointFC:
            d = TimePointFoldChange(d, TimePointFC)
        for t in t1:
            r = d.copy()
            if TreatmentFC:
                r = TreatmentFoldChange(r, TreatmentFC, t)
                c.append(r)
            else:
                r = r.loc[:, r.columns.str.contains(t)]
                c.append(r)

    elapsed = ds[0].iloc[:, 0]
    c = ConcatenateBRs(c, ftp, TimePointFC, elapsed)
    treatments = [[t] * len(lines) for t in t2] * int(c.shape[0] * (c.shape[1] - 1) / (len(lines) * len(t1)))
    t = [y for x in treatments for y in x]
    d = TransformTimeCourseMatrixForSeaborn(c, lines, TimePointFC, ylabel, t)

    if plot == "Full":
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=figsize)
        for i, line in enumerate(lines):
            x = d[d["Lines"] == line]
            if i < 5:
                sns.lineplot(x="Elapsed (h)", y=ylabel, hue="Treatments", data=x, err_style="bars", ci=68, ax=ax[0, i])
                ax[0, i].set_title(line)
                ax[0, i].set_ylabel(ylabel)
                if i > 0:
                    ax[0, i].legend().remove()
            else:
                sns.lineplot(x="Elapsed (h)", y=ylabel, hue="Treatments", data=x, err_style="bars", ci=68, ax=ax[1, i - 5])
                ax[1, i - 5].set_title(line)
                ax[1, i - 5].set_ylabel(ylabel)
                ax[1, i - 5].legend().remove()

    if plot != "Full":
        x = d[d["Lines"] == plot]
        sns.lineplot(x="Elapsed (h)", y=ylabel, hue="Treatments", data=x, err_style="bars", ci=68, ax=ax_)
        ax_.set_title(plot)
        ax_.set_ylabel(ylabel)

    if savefig:
        fig.savefig("TimeCourse.pdf", bbox_inches="tight")


def TimePointFoldChange(d, itp):
    """ Take fold-change of the time lapse data set to an initial time point  """
    for jj in range(1, d.columns.size):
        d.iloc[:, jj] /= d[d["Elapsed"] == itp].iloc[0, jj]
    return d


def TreatmentFoldChange(d, FC, treatment):
    """ Take fold-change of the time lapse data set to an initial time point  """
    fcto = d.loc[:, d.columns.str.contains(FC)]
    tr = d.loc[:, d.columns.str.contains(treatment)]
    for jj in range(0, tr.columns.size):
        tr.iloc[:, jj] /= fcto.iloc[:, jj]
    return tr


def ConcatenateBRs(c, ftp, itp, elapsed):
    """ Concatenate all BRs into the same data structure, insert time point labels, and include only desired range of data points """
    c = pd.concat(c, axis=1)
    c.insert(0, "Elapsed", elapsed)
    c = c[c["Elapsed"] <= ftp]
    c = c[c["Elapsed"] >= itp]
    return c


def TransformTimeCourseMatrixForSeaborn(x, l, itp, ylabel, treatments):
    """ Preprocess data to plot with seaborn. Returns a data frame in which each row is a data point in the plot """
    y = pd.DataFrame()
    elapsed, lines, cv = [], [], []
    for idx, row in x.iterrows():
        row = pd.DataFrame(row).T
        elapsed.extend(list(row["Elapsed"]) * (row.shape[1] - 1))
        lines.extend(list(l) * (np.int((row.shape[1] - 1) / len(l))))
        cv.extend(row.iloc[0, 1:].values)

    y["Elapsed (h)"] = elapsed
    y["Lines"] = lines
    y["Treatments"] = treatments
    y[ylabel] = cv
    return y


def FormatDf(cv, t, l, ylabel):
    """ Compute fold-change plus format for seaborn bar plot. """
    dfc = pd.DataFrame()
    dfc[ylabel] = cv
    dfc["AXL mutants Y->F"] = l
    dfc["Treatment"] = t
    return dfc


def barplot_UtErlAF154(ax, lines, ds, ftp, t1, t2, ylabel, title, TimePointFC=False, TreatmentFC=False, colors=colors):
    """ Cell viability bar plot at a specific end point across conditions, with error bars.
    Note that ds should be a list containing all biological replicates."""
    ds = FixColumnLabels(ds)
    c = []
    for d in ds:
        if TimePointFC:
            d = TimePointFoldChange(d, TimePointFC)
        for t in t1:
            r = d.copy()
            if TreatmentFC:
                r = TreatmentFoldChange(r, TreatmentFC, t)
                c.append(r)
            else:
                r = r.loc[:, r.columns.str.contains(t)]
                c.append(r)

            r.insert(0, "Elapsed", ds[0].iloc[:, 0])
            z = FormatDf(r[r["Elapsed"] == ftp].iloc[0, 1:], t, lines, ylabel)
            c.append(z)

    c = pd.concat(c)
    pal = sns.xkcd_palette(colors)
    ax = sns.barplot(
        x="AXL mutants Y->F", y=ylabel, hue="Treatment", data=c, ci=68, ax=ax, palette=pal, **{"linewidth": 0.5}, **{"edgecolor": "black"}
    )

    ax.set_title(title)
    ax.set_xticklabels(lines, rotation=90)


"""Compute fold change to itp. Then for each time point between itp and ftp inclusive, compare to UT at that time. Then plot"""


def FCvsUT_TimeCourse(ax, ds, itp, ftp, lines, tr1, treatment, title, FC=False):
    c = []
    for i in range(len(ds)):
        d = ds[i].copy()
        d = d.drop(columns=["Elapsed"])
        d.insert(0, "Elapsed", ds[0].iloc[:, 0])

        if FC:
            d = ComputeFoldChange(d, itp)
        x = fc_TvsUT_Time(d, itp, ftp, lines, tr1, treatment)
        c.append(x)
    c = pd.concat(c)
    b = sns.lineplot(x="Elapsed (h)", y="Change vs UT", hue="Lines", data=c, err_style="bars", err_kws={"capsize": 7}, ci=68, ax=ax)

    ax.set_title(title)


def fc_TvsUT_Time(d, itp, ftp, lines, tr1, treatment):
    ut = pd.concat([d.iloc[:, 0], d.loc[:, d.columns.str.contains(tr1[0])]], axis=1)
    x = pd.concat([d.iloc[:, 0], d.loc[:, d.columns.str.contains(treatment)]], axis=1)
    c = []
    for time in range(itp, ftp + 1, 3):
        ut_time = ut[ut["Elapsed"] == time].iloc[0, 1:].reset_index(drop=True)
        x_time = x[x["Elapsed"] == time].iloc[0, 1:].reset_index(drop=True)

        fc = pd.DataFrame(x_time.div(ut_time)).reset_index()
        fc["Elapsed (h)"] = time
        fc["Lines"] = lines
        fc = fc[["index", "Elapsed (h)", "Lines", fc.columns[1]]]
        fc.columns = ["index", "Elapsed (h)", "Lines", "Change vs UT"]
        c.append(fc)
    c = pd.concat(c)
    return c


def Phasenorm_Timecourse(ax, dphase, dtest, itp, ftp, treatment, lines, title, FC=False):
    c = []
    for i in range(len(dphase)):
        dp = dphase[i].copy()
        dt = dtest[i].copy()
        dp = dp.drop(columns=["Elapsed"])
        dt = dt.drop(columns=["Elapsed"])
        dp.insert(0, "Elapsed", dphase[0].iloc[:, 0])
        dt.insert(0, "Elapsed", dtest[0].iloc[:, 0])

        if FC:
            dp = ComputeFoldChange(dp, itp)
            dt = ComputeFoldChange(dt, itp)
        x = fc_ConditionvsPhase_Time(dp, dt, itp, ftp, treatment, lines)
        c.append(x)
    c = pd.concat(c)
    b = sns.lineplot(x="Elapsed (h)", y="Change vs Phase", hue="Lines", data=c, err_style="bars", err_kws={"capsize": 7}, ci=68, ax=ax)

    ax.set_title(title)


def fc_ConditionvsPhase_Time(dp, dt, itp, ftp, treatment, lines):
    dp = pd.concat([dp.iloc[:, 0], dp.loc[:, dp.columns.str.contains(treatment)]], axis=1)
    dt = pd.concat([dt.iloc[:, 0], dt.loc[:, dt.columns.str.contains(treatment)]], axis=1)
    c = []
    for time in range(itp, ftp + 1, 3):
        dp_time = dp[dp["Elapsed"] == time].iloc[0, 1:].reset_index(drop=True)
        dt_time = dt[dt["Elapsed"] == time].iloc[0, 1:].reset_index(drop=True)

        fc = pd.DataFrame(dt_time.div(dp_time)).reset_index()
        fc["Elapsed (h)"] = time
        fc["Lines"] = lines
        fc = fc[["index", "Elapsed (h)", "Lines", fc.columns[1]]]
        fc.columns = ["index", "Elapsed (h)", "Lines", "Change vs Phase"]
        c.append(fc)
    c = pd.concat(c)
    return c


# Plot Separately since makefigure can't add it as a subplot
def plotClustergram(data, title, lim=False, robust=True, figsize=(10, 10)):
    """ Clustergram plot. """
    g = sns.clustermap(data, method="complete", cmap="bwr", robust=robust, vmax=lim, vmin=-lim, yticklabels=True, figsize=figsize)
    g.fig.suptitle(title, fontsize=17)
    ax = g.ax_heatmap
    ax.set_ylabel("")


def pca_dfs(scores, loadings, df, n_components, sIDX, lIDX):
    """ build PCA scores and loadings data frames. """
    dScor = pd.DataFrame()
    dLoad = pd.DataFrame()
    for i in range(n_components):
        cpca = "PC" + str(i + 1)
        dScor[cpca] = scores[:, i]
        dLoad[cpca] = loadings[i, :]

    for j in sIDX:
        dScor[j] = list(df[j])
    dLoad[lIDX] = df.select_dtypes(include=["float64"]).columns
    return dScor, dLoad


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
    ax[0].set_title("PCA Scores")
    ax[0].set_xlabel("Principal Component 1")
    ax[0].set_ylabel("Principal Component 2")
    ax[0].axhline(y=0, color="0.25", linestyle="--")
    ax[0].axvline(x=0, color="0.25", linestyle="--")

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
        "SLC35E1 Y403-p",
    ]
    for i, name in enumerate(pn):
        p = name + " " + ps[i]
        if p in poi:
            ax[1].annotate(name, (PC1_loadings[i], PC2_loadings[i]))
    ax[1].scatter(PC1_loadings, PC2_loadings, c="darkred", linewidths=0.2, alpha=0.7)
    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("Principal Component 1")
    ax[1].set_ylabel("Principal Component 2")
    ax[1].axhline(y=0, color="0.25", linestyle="--")
    ax[1].axvline(x=0, color="0.25", linestyle="--")

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
            x=loadings["PC1"],
            y=loadings["PC2"],
            opacity=0.7,
            text=["Protein: " + loadings.index[i][0] + "  Pos: " + loadings.index[i][1] for i in range(len(loadings.index))],
            marker=dict(color="crimson", size=8, line=dict(color="black", width=1)),
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
        title_text=title,
    ),
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
    assert DupsTable.shape[0] / 2 == DupsTable_drop.shape[0]

    # Stdev of Triplicates, optionally filtering first
    StdPeptides = BuildMatrix(StdPeptides, ABC)
    TripsTable = TripsMeanAndStd(StdPeptides, list(ABC.columns[:3]))
    Stds = TripsTable.iloc[:, TripsTable.columns.get_level_values(1) == "std"]
    # Xidx = np.all(Stds.values <= 0.4, axis=1)
    # Stds = Stds.iloc[Xidx, :]

    n_bins = 10
    ax[0].hist(DupsTable_drop.iloc[:, -1], bins=n_bins, edgecolor="black", linewidth=1)
    ax[0].set_ylabel("Number of peptides")
    ax[0].set_xlabel("Pearson Correlation Coefficients (n=" + str(DupsTable_drop.shape[0]) + ")")
    textstr = "$r2$ mean = " + str(np.round(DupsTable_drop.iloc[:, -1].mean(), 2))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax[0].text(0.03, 0.96, textstr, transform=ax[0].transAxes, verticalalignment="top", bbox=props)

    ax[1].hist(Stds.mean(axis=1), bins=n_bins, edgecolor="black", linewidth=1)
    ax[1].set_ylabel("Number of peptides")
    ax[1].set_xlabel("Mean of Standard Deviations (n=" + str(Stds.shape[0]) + ")")
    textstr = "$σ$ mean = " + str(np.round(np.mean(Stds.mean(axis=1)), 2))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax[1].text(0.75, 0.96, textstr, transform=ax[1].transAxes, verticalalignment="top", bbox=props)


def plot_AllSites(ax, x, prot, title):
    """ Plot all phosphopeptides for a given protein. """
    x = x.set_index(["Gene"])
    peptides = pd.DataFrame(x.loc[prot])
    assert peptides.shape[0] > 0
    if peptides.shape[1] == 1:
        peptides = peptides.T
        d = peptides.iloc[:, 4:]
    else:
        d = peptides.select_dtypes(include=["float64"])

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
    x = x.set_index(["Gene", "Position"])
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
