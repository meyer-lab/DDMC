"""
This creates Figure 2: Evaluation of Imputating Missingness
"""
import matplotlib
import numpy as np
from scipy.stats import gmean
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..clustering import DDMC
from ..pre_processing import filter_NaNpeptides
from fancyimpute import IterativeSVD


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3), multz={0: 2})

    # Set plotting format
    matplotlib.rcParams["font.sans-serif"] = "Arial"
    sns.set(
        style="whitegrid",
        font_scale=1,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Add subplot labels
    subplotLabel(ax)

    # diagram explaining reconstruction process
    ax[0].axis("off")
    
    n_clusters = np.arange(1, 46, 45) 

    # Imputation error across Cluster numbers
    dataC_W0 = run_repeated_imputation(
        "Binomial", [0] * len(n_clusters), n_clusters=n_clusters, n_runs=1
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
    # gm["Minimum"] = np.log(data.groupby([kind]).Minimum.apply(gmean).values)
    gm["PCA"] = np.log(data.groupby([kind]).PCA.apply(gmean).values)

    gm.to_csv("WeightSearch.csv")

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
    # sns.regplot(x=kind, y="Minimum", data=gm, color="green", scatter=False, ax=ax, label="Minimum")
    sns.regplot(
        x=kind, y="PCA", data=gm, color="orange", scatter=False, ax=ax, label="PCA"
    )
    ax.set_title(title)
    ax.set_ylabel("log(MSE)—Actual vs Imputed")
    ax.legend(prop={"size": 10}, loc="upper left")
    if not legend:
        ax.legend().remove()


def run_repeated_imputation(distance_method, weights, n_clusters, n_runs=1, tmt=6):
    """Calculate missingness error across different numbers of clusters and/or weights."""
    assert len(weights) == len(n_clusters)
    X_raw = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        tmt=tmt,
    )
    # reset index
    X_raw.reset_index(drop=True, inplace=True)

    info_cols = ["Sequence", "Protein", "Gene", "Position"]
    sample_cols = [col for col in X_raw.columns if col not in info_cols]
    info = X_raw[info_cols].copy()
    X = X_raw[sample_cols].copy()

    # the condition in which each sample was collected
    sample_to_condition_df = pd.read_csv("ddmc/data/MS/CPTAC/IDtoExperiment.csv")
    assert all(
        sample_to_condition_df.iloc[:, 0] == X.columns
    ), "Sample labels don't match."
    X = X.to_numpy()
    sample_to_condition = sample_to_condition_df["Experiment (TMT10plex)"].to_numpy()
    assert X.shape[1] == sample_to_condition.size

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
        X_miss = add_missingness(X, sample_to_condition)
        baseline_imputations = [
            impute_mean(X_miss),
            impute_zero(X_miss),
            impute_min(X_miss),
            impute_pca(X, 5),
        ]
        baseline_errs = [
            imputation_error(X, X_impute) for X_impute in baseline_imputations
        ]

        for jj, cluster in enumerate(n_clusters):
            print("hi")
            df.loc[len(df)] = [
                ii,
                cluster,
                weights[jj],
                imputation_error(
                    X, impute_ddmc(X, info, cluster, weights[jj], distance_method)
                ),
                *baseline_errs,
            ]

    return df


def add_missingness(X, sample_to_experiment):
    """Remove a random TMT experiment for each peptide."""
    X = X.copy()
    for ii in range(X.shape[0]):
        tmtNum = sample_to_experiment[np.isfinite(X[ii, :])]
        X[ii, sample_to_experiment == np.random.choice(np.unique(tmtNum))] = np.nan
    return X


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


def impute_ddmc(X, info, n_clusters, weight, distance_method):
    return DDMC(info, n_clusters, weight, distance_method, max_iter=1, tol=0.1).fit(X.T).impute(X)


makeFigure()
