"""
This creates Figure 1: Phenotypic characterization of PC9 AXL mutants
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from ..motifs import MapMotifs
from ..pca import plotPCA
from ..pre_processing import y_pre, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, FixColumnLabels, CorrCoefFilter
from ..distances import DataFrameRipleysK, PlotRipleysK
import matplotlib

sns.set(color_codes=True)


path = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.max_columns", 30)

mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
all_lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
lines = ["WT", "KO", "KI", "KD", "634", "643", "698", "726", "750", "821"]
itp = 24


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((16, 12), (4, 4), multz={0: 1, 3: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in phenotype data
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")

    # AXL mutants cartoon
    ax[0].axis("off")

    # AXL expression data
    axl = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/AXLexpression.csv")
    axl = pd.melt(axl, value_vars=["AXL", "GFP"], id_vars="AXL mutants Y—>F", value_name="% Cells", var_name="Signal")
    sns.barplot(data=axl, x="AXL mutants Y—>F", y="% Cells", hue="Signal", ax=ax[1], palette=sns.xkcd_palette(["white", "darkgreen"]), **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax[1].set_title("Ectopic AXL expression")
    ax[1].legend(prop={'size': 8})

    # Migration images
    ax[2].axis("off")

    # Islands images
    ax[3].axis("off")

    # PCA phenotypes
    y = formatPhenotypesForModeling(cv, red, sw, c)
    plotPCA(ax[4:6], y, 3, ["Lines", "Treatment"], "Phenotype", hue_scores="Lines", style_scores="Treatment", legendOut=True)

    # Labels
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]

    # Cell Viability
    IndividualTimeCourses(cv, 96, all_lines, tr1, tr2, "fold-change confluency", TimePointFC=itp, TreatmentFC="-E", plot="Y698F", ax_=ax[6], ylim=[0.8, 3.5], title="Viability Y698F")
    IndividualTimeCourses(cv, 96, all_lines, tr1, tr2, "fold-change confluency", TimePointFC=itp, TreatmentFC="-E", plot="Y821F", ax_=ax[7], ylim=[0.8, 3.5], title="Viability Y821F")

    # Cell Death
    IndividualTimeCourses(red, 96, all_lines, tr1, tr2, "fold-change apoptosis (YOYO+)", TimePointFC=itp, plot="Y821F", ax_=ax[8], ylim=[0, 13], title="Death Y821F")
    IndividualTimeCourses(red, 96, all_lines, tr1, tr2, "fold-change apoptosis (YOYO+)", TimePointFC=itp, plot="Y750F", ax_=ax[9], ylim=[0, 13], title="Death Y750F")

    # Cell Migration
    t1 = ["UT", "AF", "-E", "A/E"]
    t2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]
    IndividualTimeCourses(sw, 24, all_lines, t1, t2, "RWD %", plot="Y726F", ax_=ax[10], title="Migration Y726F")
    IndividualTimeCourses(sw, 24, all_lines, t1, t2, "RWD %", plot="Y821F", ax_=ax[11], title="Migration Y821F")

    # Island Effect
    PlotRipleysK('48hrs', 'M7', ['ut', 'e', 'ae'], 6, ax=ax[12], title="Island Y726F")
    PlotRipleysK('48hrs', 'M4', ['ut', 'e', 'ae'], 6, ax=ax[13], title="Island Y750F")

    return f


def import_phenotype_data(phenotype="Cell Viability"):
    """Import all bioreplicates of a specific phenotype"""
    if phenotype == "Cell Viability":
        cv1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR1_Phase.csv")
        cv2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR2_Phase.csv")
        cv3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")
        cv4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR4_Phase.csv")
        res = FixColumnLabels([cv1, cv2, cv3, cv4])

    elif phenotype == "Cell Death":
        red1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR1_RedCount.csv")
        red2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR2_RedCount.csv")
        red3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR3_RedCount.csv")
        red4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR4_RedCount.csv")
        red4.columns = red3.columns
        res = FixColumnLabels([red1, red2, red3, red4])
        res = normalize_cellsDead_to_cellsAlive(res)

    elif phenotype == "Migration":
        sw2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR2_RWD.csv")
        sw3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR3_RWD.csv")
        sw4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR4_RWD.csv")
        res = fix_migration_columns(sw2, sw3, sw4)

    elif phenotype == "Island":
        res = DataFrameRipleysK('48hrs', mutants, ['ut', 'e', 'ae'], 6, np.linspace(1, 14.67, 1)).reset_index().set_index("Mutant")
        res.columns = ["Treatment", "Island"]

    return res


def normalize_cellsDead_to_cellsAlive(red):
    """Correct for number of alive cells to quantify dead cells"""
    cv = import_phenotype_data("Cell Viability")
    for jj in range(1, red[1].columns.size):
        red[0].iloc[:, jj] /= cv[0].iloc[:, jj]
        red[1].iloc[:, jj] /= cv[1].iloc[:, jj]
        red[2].iloc[:, jj] /= cv[2].iloc[:, jj]
        red[3].iloc[:, jj] /= cv[3].iloc[:, jj]
    return red


def formatPhenotypesForModeling(cv, red, sw, c):
    """Format and merge phenotye data sets for modeling"""
    # Cell Viability
    v_ut = y_pre(cv, "UT", 96, "Viability", all_lines, itp=itp)
    v_e = y_pre(cv, "-E", 96, "Viability", all_lines, itp=itp)
    v_ae = y_pre(cv, "A/E", 96, "Viability", all_lines, itp=itp)

    # Cell Death
    cd_ut = y_pre(red, "UT", 96, "Apoptosis", all_lines, itp=itp)
    cd_e = y_pre(red, "-E", 96, "Apoptosis", all_lines, itp=itp)
    cd_ae = y_pre(red, "A/E", 96, "Apoptosis", all_lines, itp=itp)

    # Migration
    m_ut = y_pre(sw, "UT", 10, "Migration", all_lines)
    m_e = y_pre(sw, "-E", 10, "Migration", all_lines)
    m_ae = y_pre(sw, "A/E", 10, "Migration", all_lines)
    m_ut.index = v_ut.index
    m_e.index = v_e.index
    m_ae.index = v_ae.index

    # Island
    c_ut = format_islands_byTreatments(c, "ut")
    c_e = format_islands_byTreatments(c, "e")
    c_ae = format_islands_byTreatments(c, "ae")

    # Merge and Normalize
    y_ae = pd.concat([v_ae, cd_ae["Apoptosis"], m_ae["Migration"], c_ae["Island"]], axis=1)
    y_e = pd.concat([v_e, cd_e["Apoptosis"], m_e["Migration"], c_e["Island"]], axis=1)
    y_ut = pd.concat([v_ut, cd_ut["Apoptosis"], m_ut["Migration"], c_ut["Island"]], axis=1)
    y = pd.concat([y_ut, y_e, y_ae])
    y.iloc[:, 2:] = StandardScaler().fit_transform(y.iloc[:, 2:])

    return y


def format_islands_byTreatments(island_data, treatment):
    """Find and format subset of data corresponding to each treatment"""
    X = island_data[island_data["Treatment"] == treatment]
    X = X.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    X.index = all_lines
    X = X.reset_index()
    X["Treatment"] = treatment
    return X


def fix_migration_columns(sw2, sw3, sw4):
    """Format column labels of scratch wound data"""
    cols = []
    for label in sw2.columns:
        cols.append(label.replace(" ", "-"))

    sw2.columns = cols
    sw3.columns = cols
    sw4.columns = cols
    return [sw2, sw3, sw4]


def IndividualTimeCourses(
    ds, ftp, lines, t1, t2, ylabel, TimePointFC=False, TreatmentFC=False, savefig=False, plot="Full", ax_=False, figsize=(20, 10), title=False, ylim=False
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
        if title:
            ax_.set_title(title)
        else:
            ax_.set_title(plot)
        ax_.set_ylabel(ylabel)
        ax_.legend(prop={'size': 8})
        if ylim:
            ax_.set_ylim(ylim)

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
    tr = d.loc[:, d.columns.str.contains(treatment)].copy()

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
    for _, row in x.iterrows():
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


def barplot_UtErlAF154(ax, lines, ds, ftp, t1, t2, ylabel, title, colors, TimePointFC=False, TreatmentFC=False, loc='best'):
    """ Cell viability bar plot at a specific end point across conditions, with error bars.
    Note that ds should be a list containing all biological replicates."""
    ds = FixColumnLabels(ds)
    c = []
    for d in ds:
        if TimePointFC:
            d = TimePointFoldChange(d, TimePointFC)
        for ii, t in enumerate(t1):
            r = d.copy()
            if TreatmentFC:
                r = TreatmentFoldChange(r, TreatmentFC, t)
            else:
                r = r.loc[:, r.columns.str.contains(t)]

            r.insert(0, "Elapsed", ds[0].iloc[:, 0])
            z = FormatDf(r[r["Elapsed"] == ftp].iloc[0, 1:], t2[ii], lines, ylabel)
            c.append(z.reset_index(drop=True))

    c = pd.concat(c, axis=0)
    pal = sns.xkcd_palette(colors)

    if TreatmentFC:
        c = c[~c["Treatment"].str.contains("Erlotinib")]
        ax.axhline(1, ls='--', label="Erlotinib", color="red", linewidth=1)

    ax = sns.barplot(
        x="AXL mutants Y->F", y=ylabel, hue="Treatment", data=c, ci=68, ax=ax, palette=pal, **{"linewidth": 0.5}, **{"edgecolor": "black"}
    )

    ax.set_title(title)
    ax.set_xticklabels(lines, rotation=90)
    ax.legend(prop={'size': 8}, loc=loc)


# Add clustergram to manuscript as an svg file since makefigure can't add it as a subplot object
def plotClustergram(data, title=False, lim=False, robust=True, ylabel="", yticklabels=False, xticklabels=False, figsize=(10, 10)):
    """ Clustergram plot. """
    g = sns.clustermap(data, method="centroid", cmap="bwr", robust=robust, vmax=lim, vmin=-lim, figsize=figsize, yticklabels=yticklabels, xticklabels=xticklabels)
    ax = g.ax_heatmap
    ax.set_ylabel(ylabel)


def plotVarReplicates(ax, ABC, Set_CorrCoefFilter=False, StdFilter=False):
    """ Plot variability of overlapping peptides across MS biological replicates. """
    ABC = MapMotifs(ABC, list(ABC.iloc[:, 0]))
    data_headers = list(ABC.select_dtypes(include=["float64"]).columns)
    merging_indices = list(ABC.select_dtypes(include=["object"]).columns)
    FCto = data_headers[0]
    _, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    # Correlation of Duplicates, optionally filtering first
    DupsTable = BuildMatrix(CorrCoefPeptides, ABC, data_headers)
    if Set_CorrCoefFilter:
        DupsTable = CorrCoefFilter(DupsTable, 0.5)
    DupsTable_drop = DupsTable.drop_duplicates(["Protein", "Sequence"])
    assert DupsTable.shape[0] / 2 == DupsTable_drop.shape[0]

    # Stdev of Triplicates, optionally filtering first
    StdPeptides = BuildMatrix(StdPeptides, ABC, data_headers)
    TripsTable = TripsMeanAndStd(StdPeptides, merging_indices + ["BioReps"], data_headers)
    Stds = TripsTable.iloc[:, TripsTable.columns.get_level_values(1) == "std"]
    if StdFilter:
        Xidx = np.all(Stds.values <= 0.6, axis=1)
        Stds = Stds.iloc[Xidx, :]

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


def plot_AllSites(ax, x, prot, title, ylim=False):
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
            label = positions
        else:
            label = positions[i]
        ax.plot(d.iloc[i, :], label=label, color=colors_[i])

    ax.legend(loc=0)
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    ax.set_xticklabels(lines, rotation=45)
    ax.set_ylabel("$Log2$ (p-site)")
    ax.set_title(title)
    ax.legend(prop={'size': 8})

    if ylim:
        ax.set_ylim(ylim)


def plot_IdSites(ax, x, d, title, rn=False, ylim=False, xlabels=False):
    """ Plot a set of specified p-sites. 'd' should be a dictionary werein every item is a protein-position pair. """
    n = list(d.keys())
    p = list(d.values())
    dfs = []
    for i in range(len(n)):
        x1 = x[(x["Gene"] == n[i]) & (x["Position"] == p[i])]
        dfs.append(x1.set_index(["Gene", "Position"]).select_dtypes(include=float))

    df = pd.concat(dfs)

    if rn:
        df = df.reset_index()
        df["Gene"] = rn
        df = df.set_index(["Gene", "Position"])

    data = pd.melt(frame=df.reset_index(), id_vars=["Gene", "Position"], value_vars=df.columns, var_name="Line", value_name="p-signal")
    data["GenePos"] = [g + ": " + p for g, p in zip(data["Gene"], data["Position"])]

    ax = sns.lineplot(x="Line", y="p-signal", data=data, hue="GenePos", ax=ax)

    ax.legend(loc=0)
    if xlabels:
        lines = xlabels
    else:
        lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    ax.set_xticklabels(lines, rotation=45)
    ax.set_ylabel("$Log2$ (p-site)")
    ax.set_title(title)
    ax.legend(prop={'size': 8})

    if ylim:
        ax.set_ylim(ylim)


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
