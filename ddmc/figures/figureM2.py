"""
This creates Figure 2: Evaluation of Imputating Missingness
"""

import numpy as np
from scipy.stats import gmean
import pandas as pd
import seaborn as sns
from .common import getSetup
from ..clustering import DDMC
from ..pre_processing import filter_NaNpeptides
from fancyimpute import IterativeSVD
from ddmc.datasets import CPTAC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3), multz={0: 2})

    # diagram explaining reconstruction process
    ax[0].axis("off")

    n_clusters = np.arange(1, 46, 45)

    # Imputation error across Cluster numbers
    dataC_W0 = run_repeated_imputation(
        "PAM250", [0] * len(n_clusters), n_clusters=n_clusters, n_runs=1
    )
    plot_imputation_errs(ax[1], dataC_W0, "Clusters")
    ax[1].set_ylim(10.5, 12)

    dataC_W25 = run_repeated_imputation(
        "Binomial", [100] * len(n_clusters), n_clusters=n_clusters, n_runs=1
    )
    plot_imputation_errs(ax[2], dataC_W25, "Clusters")
    ax[2].set_ylim(10.5, 12)

    dataC_W100 = run_repeated_imputation(
        "Binomial", [1000000] * len(n_clusters), n_clusters=n_clusters, n_runs=1
    )
    plot_imputation_errs(ax[3], dataC_W100, "Clusters")
    ax[3].set_ylim(10.5, 12)

    # Imputation error across different weights
    weights = [0, 100]
    dataW_2C = run_repeated_imputation(
        "Binomial", weights=weights, n_clusters=[2] * len(weights), n_runs=1
    )
    plot_imputation_errs(ax[4], dataW_2C, "Weight", legend=False)
    ax[4].set_ylim(10.5, 12)

    dataW_20C = run_repeated_imputation(
        "Binomial", weights=weights, n_clusters=[20] * len(weights), n_runs=1
    )
    plot_imputation_errs(ax[5], dataW_20C, "Weight", legend=False)
    ax[5].set_ylim(10.5, 12)

    dataW_40C = run_repeated_imputation(
        "Binomial", weights=weights, n_clusters=[40] * len(weights), n_runs=1
    )
    plot_imputation_errs(ax[6], dataW_40C, "Weight", legend=False)
    ax[6].set_ylim(10.5, 12)

    return f


def plot_imputation_errs(ax, data, kind, legend=True):
    """Plot artificial missingness error across different number of clusters or weighths."""
    if kind == "Weight":
        title = "Weight Selection"
    else:
        title = "Cluster Number Selection"

    gm = pd.DataFrame(data.groupby([kind]).DDMC.apply(gmean)).reset_index()
    gm["DDMC"] = np.log(gm["DDMC"])
    gm["Average"] = np.log(data.groupby([kind]).Average.apply(gmean).values)
    gm["Zero"] = np.log(data.groupby([kind]).Zero.apply(gmean).values)
    gm["PCA"] = np.log(data.groupby([kind]).PCA.apply(gmean).values)

    sns.regplot(
        x=kind,
        y="DDMC",
        data=gm,
        scatter_kws={"alpha": 0.25},
        color="darkblue",
        ax=ax,
        label="DDMC",
        lowess=True,
    )
    sns.regplot(
        x=kind,
        y="Average",
        data=gm,
        color="black",
        scatter=False,
        ax=ax,
        label="Average",
    )
    sns.regplot(
        x=kind, y="Zero", data=gm, color="lightblue", scatter=False, ax=ax, label="Zero"
    )
    sns.regplot(
        x=kind, y="PCA", data=gm, color="orange", scatter=False, ax=ax, label="PCA"
    )
    ax.set_title(title)
    ax.set_ylabel("log(MSE)—Actual vs Imputed")
    ax.legend(prop={"size": 10}, loc="upper left")
    if not legend:
        ax.legend().remove()


def run_repeated_imputation(distance_method, weights, n_clusters, n_runs=1):
    """Calculate missingness error across different numbers of clusters and/or weights."""
    assert len(weights) == len(n_clusters)
    cptac = CPTAC()
    p_signal = cptac.get_p_signal()
    sample_to_experiment = cptac.get_sample_to_experiment()

    df = pd.DataFrame(
        columns=[
            "N_Run",
            "Clusters",
            "Weight",
            "DDMC",
            "Average",
            "Zero",
            "Minimum",
            "PCA",
        ]
    )

    for ii in range(n_runs):
        X_miss = add_missingness(p_signal, sample_to_experiment)
        baseline_imputations = [
            impute_mean(X_miss),
            impute_zero(X_miss),
            impute_min(X_miss),
            impute_pca(X_miss, 5),
        ]
        baseline_errs = [
            imputation_error(p_signal, X_impute) for X_impute in baseline_imputations
        ]

        for jj, cluster in enumerate(n_clusters):
            df.loc[len(df)] = [
                ii,
                cluster,
                weights[jj],
                imputation_error(
                    p_signal,
                    impute_ddmc(p_signal, cluster, weights[jj], distance_method),
                ),
                *baseline_errs,
            ]
    return df


def add_missingness(p_signal, sample_to_experiment):
    """Remove a random TMT experiment for each peptide."""
    p_signal = p_signal.copy()
    for ii in range(p_signal.shape[0]):
        experiments = sample_to_experiment[np.isfinite(p_signal[ii, :])]
        p_signal[
            ii, sample_to_experiment == np.random.choice(np.unique(experiments))
        ] = np.nan
    return p_signal


def imputation_error(X, X_impute):
    # returns MSE between X and X_impute
    mse = np.sum(np.square(X - X_impute))
    assert mse != np.NaN
    return mse


def impute_zero(X):
    X = X.copy()
    X[np.isnan(X)] = 0.0
    return X


def impute_min(X):
    X = X.copy()
    np.copyto(X, np.nanmin(X, axis=0, keepdims=True), where=np.isnan(X))
    return X


def impute_mean(X):
    X = X.copy()
    np.copyto(X, np.nanmean(X, axis=0, keepdims=True), where=np.isnan(X))
    return X


def impute_pca(X, rank):
    X = X.copy()
    return IterativeSVD(rank=rank, verbose=False).fit_transform(X)


def impute_ddmc(p_signal, n_clusters, weight, distance_method):
    return (
        DDMC(n_clusters, weight, distance_method, max_iter=1)
        .fit(p_signal)
        .impute(p_signal)
    )
