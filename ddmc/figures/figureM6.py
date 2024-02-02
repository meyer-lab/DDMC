import numpy as np
import pandas as pd
import seaborn as sns
from bioinfokit import visuz
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC
from ddmc.figures.common import getSetup, plot_cluster_kinase_distances
from ddmc.logistic_regression import plotROC, plotClusterCoefficients


def makeFigure():
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1})
    cptac = CPTAC()
    p_signal = cptac.get_p_signal()
    model = DDMC(n_components=30, seq_weight=100, max_iter=10).fit(p_signal)

    # Import Genotype data
    egfrm = cptac.get_mutations(["EGFR.mutation.status"])

    # Find centers
    centers = model.transform(as_df=True).loc[egfrm.index]

    pvals = []
    centers_m = centers[egfrm]
    centers_wt = centers[~egfrm]
    for col in centers.columns:
        pvals.append(mannwhitneyu(centers_m[col], centers_wt[col])[1])
    pvals = multipletests(pvals)[1]

    # plot tumor vs nat by cluster
    df_violin = (
        centers.assign(m=egfrm)
        .reset_index()
        .melt(
            id_vars="m",
            value_vars=centers.columns,
            value_name="p-signal",
            var_name="Cluster",
        )
    )
    sns.violinplot(
        data=df_violin,
        x="Cluster",
        y="p-signal",
        hue="m",
        dodge=True,
        ax=axes[0],
        linewidth=0.25,
    )

    annotation_height = df_violin["p-signal"].max() + 0.02
    for i, pval in enumerate(pvals):
        if pval < 0.05:
            annotation = "*"
        elif pval < 0.01:
            annotation = "**"
        else:
            continue
        axes[0].text(
            i, annotation_height, annotation, ha="center", va="bottom", fontsize=10
        )

    # Normalize
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers)

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=20,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )
    plotROC(lr, centers.values, egfrm.values, cv_folds=3, title="ROC EGFRm", ax=axes[1])
    axes[1].legend(loc="lower right", prop={"size": 8})

    plotClusterCoefficients(axes[2], lr, title="")

    top_clusters = np.argsort(np.abs(lr.coef_.squeeze()))[-3:]
    #  plot predicted kinases for most weighted clusters
    distances = model.predict_upstream_kinases()[top_clusters]

    # plot Upstream Kinases
    plot_cluster_kinase_distances(
        distances, model.get_pssms(clusters=top_clusters), axes[3], num_hits=2
    )
    return f


# THIS FUNCTION IS NOT MAINTAINED
def make_EGFRvolcano_plot(centers, y):
    """Make volcano plot with differential protein expression between EGFRm and WT."""
    y = y.drop("C3N.00545")
    centers = centers.drop("C3N.00545")
    prot = (
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC_LUAD_Protein.csv")
        .iloc[:, 15:]
        .dropna()
        .drop("id.1", axis=1)
        .drop_duplicates()
    )
    prot = prot.set_index("GeneSymbol").T.sort_index().reset_index()
    prot = prot[~prot["index"].str.endswith(".N")].set_index("index")
    prot.columns.name = None

    l1 = list(centers.index)
    l2 = list(prot.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    prot = prot.drop(dif)
    assert np.all(centers.index.values == prot.index.values), "Samples don't match"

    egfr = y["EGFR.mutation.status"].replace(0, "WT")
    egfr = egfr.replace(1, "Mut")
    prot["EGFR Status"] = egfr.values
    prot = prot.set_index("EGFR Status")

    pvals = mannwhitneyu(prot.loc["Mut"], prot.loc["WT"])[1]
    pvals = multipletests(pvals)[1]

    means = prot.reset_index().groupby("EGFR Status").mean()
    fc = means.iloc[0, :] - means.iloc[1, :]

    pv = pd.DataFrame()
    pv["Gene"] = prot.columns
    pv["p-values"] = pvals
    pv["logFC"] = fc.values
    pv = pv.sort_values(by="p-values")

    visuz.gene_exp.volcano(
        df=pv,
        lfc="logFC",
        pv="p-values",
        show=True,
        geneid="Gene",
        genenames="deg",
        figtype="svg",
    )
