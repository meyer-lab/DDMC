import numpy as np
import pandas as pd
from bioinfokit import visuz
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.multitest import multipletests

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC
from ddmc.figures.common import (
    getSetup,
    plot_cluster_kinase_distances,
    plot_p_signal_across_clusters_and_binary_feature,
)
from ddmc.logistic_regression import (
    plot_roc,
    plot_cluster_regression_coefficients,
    normalize_cluster_centers,
    get_highest_weighted_clusters,
)


def makeFigure():
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1})
    cptac = CPTAC()
    p_signal = cptac.get_p_signal()
    model = DDMC(n_components=30, seq_weight=100).fit(p_signal)

    # Import Genotype data
    egfrm = cptac.get_mutations(["EGFR.mutation.status"])["EGFR.mutation.status"]

    # Find centers
    centers = model.transform(as_df=True).loc[egfrm.index]

    plot_p_signal_across_clusters_and_binary_feature(
        egfrm, centers, "egfr mutation", axes[0]
    )

    # Normalize
    centers.iloc[:, :] = normalize_cluster_centers(centers.values)

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=20,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )
    plot_roc(
        lr, centers.values, egfrm.values, cv_folds=3, title="ROC EGFRm", ax=axes[1]
    )
    axes[1].legend(loc="lower right", prop={"size": 8})

    plot_cluster_regression_coefficients(axes[2], lr, title="")

    top_clusters = get_highest_weighted_clusters(model, lr.coef_)

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
