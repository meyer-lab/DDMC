"""
This creates Figure 1: Phenotypic characterization of PC9 AXL mutants
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from ..motifs import MapMotifs
from ..pre_processing import preprocessing, y_pre, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, FixColumnLabels
from ..distances import BarPlotRipleysK, DataFrameRipleysK, PlotRipleysK

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
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

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
    ax[0].legend(prop={'size': 8})
    if legendOut:
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2, prop={'size': 8})

    # Loadings
    if isinstance(pvals, np.ndarray):
        dLoad_["p-value"] = pvals
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="p-value", style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})
    else:
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})

    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    ax[1].legend(prop={'size': 8})
    for j, txt in enumerate(dLoad_[loadings_ind]):
        ax[1].annotate(txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10)


def plotPCA_scoresORloadings(ax, d, n_components, scores_ind, loadings_ind, hue_scores=None, style_scores=None, pvals=None, style_load=None, legendOut=False, plot="scores", annotateScores=False):
    """Plot PCA scores only"""
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.select_dtypes(include=["float64"]).values)
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, scores_ind, loadings_ind)
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    if plot == "scores":
        sns.scatterplot(x="PC1", y="PC2", data=dScor_, hue=hue_scores, style=style_scores, ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})
        ax.set_title("PCA Scores")
        ax.set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
        ax.set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
        if legendOut:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2, prop={'size': 8})
        if annotateScores:
            for j, txt in enumerate(d[scores_ind[0]]):
                ax.annotate(txt, (dScor_["PC1"][j] + 0.001, dScor_["PC2"][j] + 0.001), fontsize=10)

    # Loadings
    elif plot == "loadings":
        if isinstance(pvals, np.ndarray):
            dLoad_["p-value"] = pvals
            sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="p-value", style=style_load, ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})
        else:
            sns.scatterplot(x="PC1", y="PC2", data=dLoad_, style=style_load, ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})

        ax.set_title("PCA Loadings")
        ax.set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
        ax.set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
        for j, txt in enumerate(dLoad_[loadings_ind]):
            ax.annotate(txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10)


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


def plotVarReplicates(ax, ABC, CorrCoefFilter=False, StdFilter=False):
    """ Plot variability of overlapping peptides across MS biological replicates. """
    ABC = MapMotifs(ABC, list(ABC.iloc[:, 0]))
    data_headers = list(ABC.select_dtypes(include=["float64"]).columns)
    merging_indices = list(ABC.select_dtypes(include=["object"]).columns)
    FCto = data_headers[0]
    _, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    # Correlation of Duplicates, optionally filtering first
    DupsTable = BuildMatrix(CorrCoefPeptides, ABC, data_headers, FCto)
    if CorrCoefFilter:
        DupsTable = CorrCoefFilter(DupsTable)
    DupsTable_drop = DupsTable.drop_duplicates(["Protein", "Sequence"])
    assert DupsTable.shape[0] / 2 == DupsTable_drop.shape[0]

    # Stdev of Triplicates, optionally filtering first
    StdPeptides = BuildMatrix(StdPeptides, ABC, data_headers, FCto)
    TripsTable = TripsMeanAndStd(StdPeptides, merging_indices + ["BioReps"], data_headers)
    Stds = TripsTable.iloc[:, TripsTable.columns.get_level_values(1) == "std"]
    if StdFilter:
        Xidx = np.all(Stds.values <= 0.4, axis=1)
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


def plot_IdSites(ax, x, d, title, rn=False, ylim=False):
    """ Plot a set of specified p-sites. 'd' should be a dictionary werein every item is a protein-position pair. """
    x = x.set_index(["Gene", "Position"])
    n = list(d.keys())
    p = list(d.values())
    dfs = []
    for i in range(len(n)):
        dfs.append(x.loc[n[i], p[i]].select_dtypes(include=float))

    df = pd.concat(dfs)

    if rn:
        df = df.reset_index()
        df["Gene"] = rn
        df = df.set_index(["Gene", "Position"])

    data = pd.melt(frame=df.reset_index(), id_vars=["Gene", "Position"], value_vars=df.columns, var_name="Line", value_name="p-signal")
    data["GenePos"] = [g + ": " + p for g, p in zip(data["Gene"], data["Position"])]

    ax = sns.lineplot(x="Line", y="p-signal", data=data, hue="GenePos", ax=ax)

    ax.legend(loc=0)
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
