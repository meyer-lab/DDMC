import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, select_peptide_subset
from ddmc.figures.common import getSetup, plot_cluster_kinase_distances
from ddmc.logistic_regression import plotClusterCoefficients, plotROC


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

    # get p value of tumor vs NAT for each cluster
    pvals = []
    centers_tumor = centers[is_tumor]
    centers_nat = centers[~is_tumor]
    for col in centers.columns:
        pvals.append(mannwhitneyu(centers_tumor[col], centers_nat[col])[1])
    pvals = multipletests(pvals)[1]

    # run PCA on cluster centers
    pca = PCA(n_components=2)
    scores = pca.fit_transform(centers)  # sample by PCA component
    loadings = pca.components_  # PCA component by cluster
    variance_explained = np.round(pca.explained_variance_ratio_, 2)

    # plot scores
    sns.scatterplot(
        x=scores[:, 0],
        y=scores[:, 1],
        hue=is_tumor,
        ax=axes[1],
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    axes[1].legend(loc="lower left", prop={"size": 9}, title="Tumor", fontsize=9)
    axes[1].set_title("PCA Scores")
    axes[1].set_xlabel(
        "PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10
    )
    axes[1].set_ylabel(
        "PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10
    )

    # plot loadings
    sns.scatterplot(
        x=loadings[0],
        y=loadings[1],
        ax=axes[2],
        hue=pvals < 0.01,
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    axes[2].set_title("PCA Loadings")
    axes[2].set_xlabel(
        "PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10
    )
    axes[2].set_ylabel(
        "PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10
    )
    axes[2].legend(title="p < 0.01", prop={"size": 8})
    for j, txt in enumerate(centers.columns):
        axes[2].annotate(
            txt, (loadings[0][j] + 0.001, loadings[1][j] + 0.001), fontsize=10
        )

    # plot tumor vs nat by cluster
    df_violin = (
        centers.assign(is_tumor=is_tumor)
        .reset_index()
        .melt(
            id_vars="is_tumor",
            value_vars=centers.columns,
            value_name="p-signal",
            var_name="Cluster",
        )
    )
    sns.violinplot(
        data=df_violin,
        x="Cluster",
        y="p-signal",
        hue="is_tumor",
        dodge=True,
        ax=axes[3],
        linewidth=0.25,
    )
    axes[3].legend(prop={"size": 8})

    annotation_height = df_violin["p-signal"].max() + 0.02
    for i, pval in enumerate(pvals):
        if pval < 0.05:
            annotation = "*"
        elif pval < 0.01:
            annotation = "**"
        else:
            continue
        axes[3].text(
            i, annotation_height, annotation, ha="center", va="bottom", fontsize=10
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
    plotROC(lr, centers.values, is_tumor, cv_folds=4, return_mAUC=False, ax=axes[4])

    plotClusterCoefficients(axes[5], lr)

    top_clusters = np.argsort(np.abs(lr.coef_.squeeze()))[-3:]

    # plot predicted kinases for most weighted clusters
    distances = model.predict_upstream_kinases()[top_clusters]

    plot_cluster_kinase_distances(
        distances, model.get_pssms(clusters=top_clusters), axes[6], num_hits=2
    )
    return f
