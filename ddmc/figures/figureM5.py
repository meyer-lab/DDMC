import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, select_peptide_subset
from ddmc.figures.common import (
    getSetup,
    plot_cluster_kinase_distances,
    get_pvals_across_clusters,
    plot_p_signal_across_clusters_and_binary_feature,
    plot_pca_on_cluster_centers,
)
from ddmc.logistic_regression import plot_cluster_regression_coefficients, plot_roc


def makeFigure():
    axes, f = getSetup((11, 10), (3, 3), multz={0: 1, 4: 1})
    cptac = CPTAC()
    p_signal = select_peptide_subset(cptac.get_p_signal(), keep_ratio=0.1)
    model = DDMC(n_components=30, seq_weight=100, max_iter=5).fit(p_signal)

    centers = model.transform(as_df=True)
    centers.loc[:, :] = StandardScaler(with_std=False).fit_transform(centers.values)
    is_tumor = cptac.get_tumor_or_nat(centers.index)

    # first plot heatmap of clusters
    # lim = 1.5
    # sns.clustermap(centers.set_index("Type").T, method="complete", cmap="bwr", vmax=lim, vmin=-lim,  figsize=(15, 9)) Run in notebook and save as svg
    axes[0].axis("off")

    plot_pca_on_cluster_centers(
        centers,
        axes[1:3],
        hue_scores=is_tumor,
        hue_scores_title="Tumor?",
        hue_loadings=get_pvals_across_clusters(is_tumor, centers) < 0.01,
        hue_loadings_title="p < 0.01",
    )

    plot_p_signal_across_clusters_and_binary_feature(
        is_tumor, centers, "is_tumor", axes[4]
    )

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=3,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="elasticnet",
        l1_ratios=[0.85],
        class_weight="balanced",
    )
    plot_roc(lr, centers.values, is_tumor, cv_folds=4, return_mAUC=False, ax=axes[4])

    plot_cluster_regression_coefficients(axes[5], lr)

    top_clusters = np.argsort(np.abs(lr.coef_.squeeze()))[-3:]

    # plot predicted kinases for most weighted clusters
    distances = model.predict_upstream_kinases()[top_clusters]

    plot_cluster_kinase_distances(
        distances, model.get_pssms(clusters=top_clusters), axes[6], num_hits=2
    )
    return f
