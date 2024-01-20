""" PCA functions """

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def pca_dfs(scores, loadings, df, n_components, sIDX, lIDX):
    """build PCA scores and loadings data frames."""
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


def plotPCA(
    ax,
    d,
    n_components,
    scores_ind,
    loadings_ind,
    hue_scores=None,
    style_scores=None,
    pvals=None,
    style_load=None,
    legendOut=False,
    quadrants=True,
):
    """Plot PCA scores and loadings."""
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.select_dtypes(include=["float64"]))
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, scores_ind, loadings_ind)
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    sns.scatterplot(
        x="PC1",
        y="PC2",
        data=dScor_,
        hue=hue_scores,
        style=style_scores,
        ax=ax[0],
        **{"linewidth": 0.5, "edgecolor": "k"}
    )
    ax[0].set_title("PCA Scores")
    ax[0].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[0].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    ax[0].legend(prop={"size": 8})
    if legendOut:
        ax[0].legend(
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0,
            labelspacing=0.2,
            prop={"size": 8},
        )

    # Loadings
    if isinstance(pvals, np.ndarray):
        dLoad_["p-value"] = pvals
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=dLoad_,
            hue="p-value",
            style=style_load,
            ax=ax[1],
            **{"linewidth": 0.5, "edgecolor": "k"}
        )
    else:
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=dLoad_,
            style=style_load,
            ax=ax[1],
            **{"linewidth": 0.5, "edgecolor": "k"}
        )

    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    ax[1].legend(prop={"size": 8})
    for j, txt in enumerate(dLoad_[loadings_ind]):
        ax[1].annotate(
            txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10
        )

    if quadrants:
        ax[0].axhline(0, ls="--", color="lightgrey")
        ax[0].axvline(0, ls="--", color="lightgrey")
        ax[1].axhline(0, ls="--", color="lightgrey")
        ax[1].axvline(0, ls="--", color="lightgrey")
