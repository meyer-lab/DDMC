"""
This creates Figure 2: Evaluation of Imputating Missingness
"""
import numpy as np
from scipy.stats import gmean
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides
from fancyimpute import IterativeSVD


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3), multz={0: 2})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # diagram explaining reconstruction process
    ax[0].axis("off")

    # Imputation error across Cluster numbers
    dataC_W0 = ErrorAcross("Binomial", [10] * 12, n_clusters=np.arange(1, 37, 3), n_runs=3)
    dataC_W25 = ErrorAcross("Binomial", [25] * 12, n_clusters=np.arange(1, 37, 3), n_runs=3)
    dataC_W100 = ErrorAcross("Binomial", [100] * 12, n_clusters=np.arange(1, 37, 3), n_runs=3)
    plotErrorAcrossNumberOfClustersOrWeights(ax[1], dataC_W0, "Clusters")
    print("W0 complete.")
    plotErrorAcrossNumberOfClustersOrWeights(ax[2], dataC_W25, "Clusters")
    print("W25 complete.")
    plotErrorAcrossNumberOfClustersOrWeights(ax[3], dataC_W100, "Clusters")
    print("W100 complete.")

    # # Imputation error across different Weights
    dataW_10C = ErrorAcross("Binomial", weights=np.arange(0, 55, 5), n_clusters=[10] * 11, n_runs=3)
    dataW_20C = ErrorAcross("Binomial", weights=np.arange(0, 55, 5), n_clusters=[20] * 11, n_runs=3)
    dataW_30C = ErrorAcross("Binomial", weights=np.arange(0, 55, 5), n_clusters=[30] * 11, n_runs=3)
    plotErrorAcrossNumberOfClustersOrWeights(ax[4], dataW_10C, "Weight", legend=False)
    print("10C complete.")
    plotErrorAcrossNumberOfClustersOrWeights(ax[5], dataW_20C, "Weight", legend=False)
    print("20C complete.")
    plotErrorAcrossNumberOfClustersOrWeights(ax[6], dataW_30C, "Weight", legend=False)
    print("30C complete.")

    return f


def plotErrorAcrossNumberOfClustersOrWeights(ax, data, kind, legend=True):
    """Plot artificial missingness error across different number of clusters or weighths."""
    if kind == "Weight":
        title = "Weight Selection"
    else:
        title = "Cluster Number Selection"

    gm = pd.DataFrame(data.groupby([kind]).DDMC.apply(gmean)).reset_index()
    gm["DDMC"] = np.log(gm["DDMC"])
    gm["Average"] = np.log(data.groupby([kind]).Average.apply(gmean).values)
    gm["Zero"] = np.log(data.groupby([kind]).Zero.apply(gmean).values)
    gm["Minimum"] = np.log(data.groupby([kind]).Minimum.apply(gmean).values)
    gm["PCA"] = np.log(data.groupby([kind]).PCA.apply(gmean).values)

    sns.regplot(x=kind, y="DDMC", data=gm, scatter_kws={'alpha': 0.25}, color="darkblue", ax=ax, label="DDMC")
    sns.regplot(x=kind, y="Average", data=gm, color="black", scatter=False, ax=ax, label="Average")
    sns.regplot(x=kind, y="Zero", data=gm, color="lightblue", scatter=False, ax=ax, label="Zero")
    sns.regplot(x=kind, y="Minimum", data=gm, color="green", scatter=False, ax=ax, label="Minimum")
    sns.regplot(x=kind, y="PCA", data=gm, color="orange", scatter=False, ax=ax, label="PCA")
    ax.set_xticks(list(set(gm[kind])))
    ax.set_title(title)
    ax.set_ylabel("log(MSE)—Actual vs Imputed")
    ax.legend(prop={'size': 10}, loc='upper left')
    if not legend:
        ax.legend().remove()


def ErrorAcross(distance_method, weights, n_clusters, n_runs=1, tmt=6):
    """ Calculate missingness error across different number of clusters. """
    assert len(weights) == len(n_clusters)
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=tmt)
    X.index = np.arange(X.shape[0])
    md = X.copy()
    info = md.select_dtypes(include=['object'])
    X = X.select_dtypes(include=['float64'])
    StoE = pd.read_csv("msresist/data/MS/CPTAC/IDtoExperiment.csv")
    assert all(StoE.iloc[:, 0] == X.columns), "Sample labels don't match."
    X = X.to_numpy()
    tmtIDX = StoE["Experiment (TMT10plex)"].to_numpy()
    assert X.shape[1] == tmtIDX.size

    df = pd.DataFrame(columns=["N_Run", "Clusters", "Weight", "DDMC", "Average", "Zero", "Minimum", "PCA"])

    for ii in range(n_runs):
        Xmiss = IncorporateMissingValues(X, tmtIDX)
        baseline_errors = ComputeBaselineErrors(X, Xmiss)

        for jj, cluster in enumerate(n_clusters):
            model = MassSpecClustering(info, cluster, weights[jj], distance_method).fit(Xmiss.T)
            eDDMC = np.nansum(np.square(X - model.impute(Xmiss)))
            dfs = pd.Series([ii, cluster, weights[jj], eDDMC, *baseline_errors], index=df.columns)
            df = df.append(dfs, ignore_index=True)

    return df


def IncorporateMissingValues(X, tmtIDX):
    """ Remove a random TMT experiment for each peptide. """
    X = X.copy()
    for ii in range(X.shape[0]):
        tmtNum = tmtIDX[np.isfinite(X[ii, :])]
        X[ii, tmtIDX == np.random.choice(np.unique(tmtNum))] = np.nan
    return X


def ComputeBaselineErrors(X, d, ncomp=5):
    """ Compute error between baseline methods (i.e. average signal, minimum signal, zero, and PCA) and real value. """
    dmean = d.copy()
    np.copyto(dmean, np.nanmean(d, axis=0, keepdims=True), where=np.isnan(d))
    emean = np.nansum(np.square(X - dmean))

    dmin = d.copy()
    np.copyto(dmin, np.nanmin(d, axis=0, keepdims=True), where=np.isnan(d))
    emin = np.nansum(np.square(X - dmin))

    dzero = d.copy()
    np.copyto(dzero, 0.0, where=np.isnan(d))
    ezero = np.nansum(np.square(X - dzero))

    dpca = IterativeSVD(rank=ncomp, verbose=False).fit_transform(d.copy())
    epca = np.nansum(np.square(X - dpca))

    return emean, ezero, emin, epca
