"""
This creates Figure 7: Tumor infiltrating immune cells
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from .figureM5 import build_pval_matrix, calculate_mannW_pvals, plot_clusters_binaryfeatures, plotPeptidesByFeature
from .figure2 import plotPCA, plotDistanceToUpstreamKinase
from ..logistic_regression import plotROC, plotClusterCoefficients
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 13), (4, 3), multz={0: 1, 3: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # Import DDMC clusters
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.loc[~centers["Patient_ID"].str.endswith(".N"), :].sort_values(by="Patient_ID").set_index("Patient_ID")

    # Import Cold-Hot Tumor data
    cent1, y = FormatXYmatrices(centers.copy())

    # Normalize
    cent1 = cent1.T
    cent1.iloc[:, :] = StandardScaler(with_std=False).fit_transform(cent1.iloc[:, :])
    cent1 = cent1.T

    # Hypothesis Testing
    cent1["TI"] = y.values
    pvals = calculate_mannW_pvals(cent1, "TI", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    cent1["TI"] = cent1["TI"].replace(0, "CTE")
    cent1["TI"] = cent1["TI"].replace(1, "HTE")
    plot_clusters_binaryfeatures(cent1, "TI", ax[0], pvals=pvals, loc="lower left")

    # Logistic Regression
    lr = LogisticRegressionCV(cv=5, solver="saga", max_iter=100000, tol=1e-10, n_jobs=-1, penalty="l2", class_weight="balanced")
    plotROC(ax[1], lr, cent1.iloc[:, :-1].values, y, cv_folds=4, title="ROC TI")
    plotClusterCoefficients(ax[2], lr.fit(cent1.iloc[:, :-1], y.values), title="TI weights")

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [6, 17, 20, 21], ax[3], num_hits=2)

    # GO
    plot_ImmuneGOs(6, ax[4], title="GO Cluster 6", n=5, max_width=20)
    plot_ImmuneGOs(17, ax[5], title="GO Cluster 17", n=5, loc='lower left')
    plot_ImmuneGOs(20, ax[6], title="GO Cluster 20", n=7, loc='lower left', max_width=30)

    # Representative Peptides Cluster 6
    y.index = y.index.rename("Sample.ID")
    y = y.reset_index().sort_values(by="Sample.ID")
    X["cluster"] = model.labels()
    c6 = X[X["cluster"] == 6].drop("cluster", axis=1)
    c6 = c6.loc[:, ~c6.columns.str.endswith(".N")]
    d = {"PAK1":"Y142-p", "ABL1": "S559-p", "DOCK11":"S306-p", "DOCK10": "S1289-p;S1292-p", "LCK":"Y192-p", "SYK":"S297-p", "TSC1":"Y508-p", "RC3H2": "S1017-p", "CTNNB1":"S552-p"} #B cell homeo PAK-DOCK10; T cell diff the rest
    plotPeptidesByFeature(c6, y, d, ["Infiltration Status", "HTE", "CTE"], ax[7], title="Cluster 6", TwoCols=True, legend_size=7)

    # Representative Peptides Cluster 17
    c17 = X[X["cluster"] == 17].drop("cluster", axis=1)
    c17 = c17.loc[:, ~c17.columns.str.endswith(".N")]
    d = {"CD44":"S697-p", "SDK1":"T2111-p", "PRKCD":"S130-p", "PLD1": "T495-p", "CAPN1":"T89-p", "GSTP1":"T35-p"}
    plotPeptidesByFeature(c17, y, d, ["Infiltration Status", "HTE", "CTE"], ax[8], title="Cluster 17")

    # Representative Peptides Cluster 20
    c20 = X[X["cluster"] == 20].drop("cluster", axis=1)
    c20 = c20.loc[:, ~c20.columns.str.endswith(".N")]
    d = {"RCAN1":"S210-p", "NFATC3":"S366-p", "EP300":"S1716-p", "HSP90AA1":"S623-p", "TAB3":"S80-p", "LYN":"Y316-p", "SDK1":"Y2096-p"}
    plotPeptidesByFeature(c20, y, d, ["Infiltration Status", "HTE", "CTE"], ax[9], title="Cluster 20")

    return f


def FormatXYmatrices(centers):
    """Make sure Y matrix has the same matching samples has the signaling centers"""
    y = pd.read_csv("msresist/data/MS/CPTAC/Hot_Cold.csv").dropna(axis=1).sort_values(by="Sample ID")
    y = y.loc[~y["Sample ID"].str.endswith(".N"), :].set_index("Sample ID")
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)

    # Transform to binary
    y = y.replace("Cold-tumor enriched", 0)
    y = y.replace("Hot-tumor enriched", 1)
    y = np.squeeze(y)

    # Remove NAT-enriched samples
    centers = centers.drop(y[y == "NAT enriched"].index)
    y = y.drop(y[y == "NAT enriched"].index).astype(int)
    assert all(centers.index.values == y.index.values), "Samples don't match"
    return centers, y


def plot_ImmuneGOs(cluster, ax, title=False, max_width=25, n=False, loc='best'):
    """Plot immune-related GO"""
    go = pd.read_csv("msresist/data/cluster_analysis/CPTAC_GO_C" + str(cluster) + ".csv")
    im = go[go["GO biological process complete"].str.contains("immune")]
    tc = go[go["GO biological process complete"].str.contains("T cell")]
    bc = go[go["GO biological process complete"].str.contains("B cell")]
    X = pd.concat([im, tc, bc])
    X = X[["GO biological process complete", "upload_1 (fold Enrichment)", "upload_1 (over/under)"]]
    if n:
        X = X.iloc[:n, :]
    X.columns = ["Biological process", "Fold Enrichment", "over/under"]
    X = X[X["Fold Enrichment"] != ' < 0.01']
    X["Fold Enrichment"] = X["Fold Enrichment"].astype(float)
    X["Biological process"] = [s.split("(GO")[0] for s in X["Biological process"]]
    X = X.sort_values(by="Fold Enrichment", ascending=False)
    sns.barplot(data=X, y="Biological process", x="Fold Enrichment", ax=ax, hue="over/under", orient="h", color="black", **{"linewidth": 2}, **{"edgecolor": "black"})
    ax.set_yticklabels(textwrap.fill(x.get_text(), max_width) for x in ax.get_yticklabels())
    if title:
        ax.set_title(title)
    else:
        ax.set_title("GO")
    ax.legend(prop={'size': 10}, loc=loc)
