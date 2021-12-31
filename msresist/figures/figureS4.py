"""
This creates Supplemental Figure 4: Specific phosphosites.
"""

import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np 
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from bioinfokit import visuz
from .common import subplotLabel, getSetup
from ..pre_processing import preprocessing


lines = ["WT", "KO", "KD", "KI", "634", "643", "698", "726", "750", "821"]

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 4), (1, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in Mass Spec data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_col=True)

    # RTKs
    kinase_heatmap(X, [("EGFR", "Y1197-p"),
            ("EGFR", "Y1172-p"),
            ("EPHA2", "T771-p"),
            ("EPHA2", "Y772-p"),
            ("ERBB2", "Y877-p"),
            ("ERBB2", "Y1248-p"),
            ("ERBB3", "Y1328-p"),
            ("MET", "S988-p")], ax=ax[0], FC=False)
    ax[0].set_title("Receptor Tyrosine Kinases")

    # Adapters
    kinase_heatmap(X, [("GRB2", "Y160-p"),
            ("SOS1", "Y1196-p"),
            ("DAPP1", "Y139-p"),
            ("SHB", "Y268-p"),
            ("SHC1", "S426-p"),
            ("CRK", "Y251-p"),
            ("EPS8", "Y485-p")], ax=ax[1], FC=False)
    ax[1].set_title("RTK adapters")

    # Kinases
    ax[2].axis("off")

    return f



def kinase_heatmap(X, prot_dict, ax, FC=False):
    """ Make a heatmap out of a dictionary wih p-sites """
    out = []
    for p, s in prot_dict:
        out.append(X.set_index(["Gene", "Position"]).loc[p, s])
    out = pd.concat(out).select_dtypes(include=[float])
    if FC:
        for ii in range(out.shape[1]):
            out.iloc[:, ii] /= out.iloc[:, 0]
    sns.heatmap(out, cmap="bwr", ax=ax)
    ax.set_xticklabels(lines)
    ax.set_xlabel("AXL Y—>F mutants")
    ax.set_ylabel("")


def kinases_clustermap(X):
    """Clustermap of all kinases showing a minimum variance acros mutants"""
    k = X[X["Protein"].str.contains("kinase")]
    XIDX = np.any(k.iloc[:, 8:-1] <= -0.5, axis=1) | np.any(k.iloc[:, 8:-1] >= 0.5, axis=1)
    k = k.iloc[list(XIDX), :].set_index(["Gene", "Position"]).select_dtypes(include=[float])
    k = k.drop("AXL")
    sns.clustermap(k, cmap="bwr", xticklabels=lines)


def AXL_volcanoplot(X):
    """AXL vs No AXL volcano plot"""
    axl_in = X[["PC9 A", "KI A"]].values
    axl_out = X[["KO A", "Kd A"]].values
    pvals = f_oneway(axl_in, axl_out, axis=1)[1]
    pvals = multipletests(pvals)[1]
    fc = axl_in.mean(axis=1) - axl_out.mean(axis=1)
    pv = pd.DataFrame()
    pv["Peptide"] = [g + ";" + p for g, p in list(zip(X["Gene"], X["Position"]))]
    pv["logFC"] = fc
    pv["p-values"] = pvals
    pv = pv.sort_values(by="p-values")
    visuz.GeneExpression.volcano(df=pv, lfc='logFC', pv='p-values', show=True, geneid="Peptide", lfc_thr=(0.5, 0.5), genenames="deg", color=("#00239CFF", "grey", "#E10600FF"), figtype="svg",  gstyle=2, axtickfontname='Arial')
