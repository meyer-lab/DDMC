import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC
from ddmc.figures.common import plot_cluster_kinase_distances, getSetup
from ddmc.logistic_regression import plotROC, plotClusterCoefficients

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1, 4: 1})
    cptac = CPTAC()
    is_hot = cptac.get_hot_cold_labels()
    p_signal = cptac.get_p_signal()
    model = DDMC(n_components=30, seq_weight=100, max_iter=10, random_state=5).fit(
        p_signal
    )
    assert (
        not model.has_empty_clusters()
    ), "This plot assumes that every cluster will have at least one peptide. Please rerun with fewer components are more peptides."

    centers = model.transform(as_df=True).loc[is_hot.index]

    pvals = []
    centers_m = centers[is_hot]
    centers_wt = centers[~is_hot]
    for col in centers.columns:
        pvals.append(mannwhitneyu(centers_m[col], centers_wt[col])[1])
    pvals = multipletests(pvals)[1]

    # plot tumor vs nat by cluster
    df_violin = (
        centers.assign(is_hot=is_hot)
        .reset_index()
        .melt(
            id_vars="is_hot",
            value_vars=centers.columns,
            value_name="p-signal",
            var_name="Cluster",
        )
    )

    sns.violinplot(
        data=df_violin,
        x="Cluster",
        y="p-signal",
        hue="is_hot",
        dodge=True,
        ax=axes[0],
        linewidth=0.25,
    )

    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers)
    lr = LogisticRegressionCV(
        cv=3, solver="saga", n_jobs=1, penalty="l1", max_iter=10000
    )
    plotROC(
        lr, centers.values, is_hot.values, cv_folds=3, title="ROC TI", return_mAUC=True
    )
    plotClusterCoefficients(axes[1], lr, title="")

    top_clusters = np.argsort(np.abs(lr.coef_.squeeze()))[-3:]

    #  plot predicted kinases for most weighted clusters
    distances = model.predict_upstream_kinases()[top_clusters]

    # plot upstream Kinases
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
