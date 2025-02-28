import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from sklearn.linear_model import LogisticRegressionCV

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, select_peptide_subset
from ddmc.figures.common import (
    plot_cluster_kinase_distances,
    getSetup,
    plot_p_signal_across_clusters_and_binary_feature,
)
from ddmc.logistic_regression import (
    plot_roc,
    plot_cluster_regression_coefficients,
    normalize_cluster_centers,
    get_highest_weighted_clusters, 
)


def makeFigure():
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1, 4: 1})
    cptac = CPTAC()
    is_hot = cptac.get_hot_cold_labels()
    p_signal = cptac.get_p_signal()
    model = DDMC(n_components=30, seq_weight=100, random_state=5).fit(p_signal)

    centers = model.transform(as_df=True).loc[is_hot.index]
    centers.iloc[:, :] = normalize_cluster_centers(centers.values)

    plot_p_signal_across_clusters_and_binary_feature(is_hot, centers, "is_hot", axes[0])

    n_cv = 15
    lr = LogisticRegressionCV(
        cv=n_cv, solver="saga", n_jobs=1, penalty="l1", max_iter=10000
    )
    plot_roc(
        lr, centers.values, is_hot.values, cv_folds=n_cv, title="ROC TI", return_mAUC=True
    )
    plot_cluster_regression_coefficients(axes[1], lr, title="")

    top_clusters = get_highest_weighted_clusters(model, lr.coef_)

    distances = model.predict_upstream_kinases()[top_clusters]

    plot_cluster_kinase_distances(
        distances, model.get_pssms(clusters=top_clusters), axes[2], num_hits=2
    )
    return f


def plot_ImmuneGOs(cluster, ax, title=False, max_width=25, n=False, loc="best"):
    # THIS FUNCION IS NOT MAINTAINED
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
