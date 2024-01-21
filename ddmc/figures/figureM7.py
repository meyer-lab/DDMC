"""
This creates Figure 7: Tumor infiltrating immune cells
"""

import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from .common import getSetup, plotDistanceToUpstreamKinase
from .figureM5 import (
    build_pval_matrix,
    calculate_mannW_pvals,
    plot_clusters_binaryfeatures,
)
from ..clustering import DDMC
from ..logistic_regression import plotROC, plotClusterCoefficients
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((11, 7), (2, 3), multz={0: 1})

    # Import signaling data
    X = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        tmt=2,
    )
    d = X.select_dtypes(include=[float]).T
    i = X["Sequence"]

    # Fit DDMC
    model = DDMC(
        i, n_components=30, SeqWeight=100, distance_method="Binomial", random_state=5
    ).fit(d)

    X = pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = (
        centers.loc[~centers["Patient_ID"].str.endswith(".N"), :]
        .sort_values(by="Patient_ID")
        .set_index("Patient_ID")
    )
    centers = centers.drop(
        [14, 24], axis=1
    )  # Drop clusters 14&24, contain only 1 peptide

    # Import Cold-Hot Tumor data
    cent1, y = FormatXYmatrices(centers.copy())

    # Normalize
    cent1 = cent1.T
    cent1.iloc[:, :] = StandardScaler(with_std=False).fit_transform(cent1.iloc[:, :])
    cent1 = cent1.T

    # Hypothesis Testing
    cent1["TI"] = y.values
    pvals = calculate_mannW_pvals(cent1, "TI", 1, 0)
    pvals = build_pval_matrix(model.n_components, pvals)
    cent1["TI"] = cent1["TI"].replace(0, "CTE")
    cent1["TI"] = cent1["TI"].replace(1, "HTE")
    plot_clusters_binaryfeatures(cent1, "TI", ax[0], pvals=pvals, loc="lower left")
    ax[0].legend(loc="lower left", prop={"size": 10})

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=15, solver="saga", n_jobs=-1, penalty="l1", max_iter=10000
    )
    plotROC(ax[1], lr, cent1.iloc[:, :-1].values, y, cv_folds=4, title="ROC TI")
    plotClusterCoefficients(
        ax[2], lr.fit(cent1.iloc[:, :-1], y.values), title="TI weights"
    )

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [17, 20, 21], ax[3], num_hits=3)

    return f


def FormatXYmatrices(centers):
    """Make sure Y matrix has the same matching samples has the signaling centers"""
    y = (
        pd.read_csv("ddmc/data/MS/CPTAC/Hot_Cold.csv")
        .dropna(axis=1)
        .sort_values(by="Sample ID")
    )
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


def plot_ImmuneGOs(cluster, ax, title=False, max_width=25, n=False, loc="best"):
    """Plot immune-related GO"""
    go = pd.read_csv("ddmc/data/cluster_analysis/CPTAC_GO_C" + str(cluster) + ".csv")
    im = go[go["GO biological process complete"].str.contains("immune")]
    tc = go[go["GO biological process complete"].str.contains("T cell")]
    bc = go[go["GO biological process complete"].str.contains("B cell")]
    X = pd.concat([im, tc, bc])
    X = X[
        [
            "GO biological process complete",
            "upload_1 (fold Enrichment)",
            "upload_1 (over/under)",
        ]
    ]
    if n:
        X = X.iloc[:n, :]
    X.columns = ["Biological process", "Fold Enrichment", "over/under"]
    X = X[X["Fold Enrichment"] != " < 0.01"]
    X["Fold Enrichment"] = X["Fold Enrichment"].astype(float)
    X["Biological process"] = [s.split("(GO")[0] for s in X["Biological process"]]
    X = X.sort_values(by="Fold Enrichment", ascending=False)
    sns.barplot(
        data=X,
        y="Biological process",
        x="Fold Enrichment",
        ax=ax,
        hue="over/under",
        orient="h",
        color="black",
        **{"linewidth": 2},
        **{"edgecolor": "black"}
    )
    ax.set_yticklabels(
        textwrap.fill(x.get_text(), max_width) for x in ax.get_yticklabels()
    )
    if title:
        ax.set_title(title)
    else:
        ax.set_title("GO")
    ax.legend(prop={"size": 10}, loc=loc)
