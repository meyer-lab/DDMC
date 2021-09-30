"""
This creates Figure 2: Evaluation of Imputating Missingness
"""
import glob
import pickle
import random
import numpy as np
from scipy.stats import gmean
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.multivariate.pca import PCA
from .common import subplotLabel, getSetup
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides, FindIdxValues
from ..binomial import Binomial
from ..pam250 import PAM250


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 2), multz={0: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # diagram explaining reconstruction process
    ax[0].axis("off")

    # Imputation error across Cluster numbers
    # plotErrorAcrossNumberOfClustersOrWeights(ax[1], "Clusters", "Binomial")

    # Imputation error across different Weights
    # plotErrorAcrossNumberOfClustersOrWeights(ax[5], "Weight", "Binomial", legend=False)

    return f


def plotMissingnessDensity(ax, d):
    """Plot amount of missingness per peptide."""
    p_nan_counts = []
    for i in range(d.shape[1]):
        p_nan_counts.append(np.count_nonzero(np.isnan(d[i])))

    sns.distplot(p_nan_counts, 10, ax=ax)
    ax.set_title("Missingness distribution in LUAD")
    ax.set_ylabel("Density")
    ax.set_xlabel("Number of missing observations per peptide")

    # Add Mean
    textstr = "$u$ = " + str(np.round(np.mean(p_nan_counts), 1))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax.text(0.015, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)


def plotErrorAcrossNumberOfClustersOrWeights(ax, kind, distance_method, weight=10, ncl=20, n_runs=5, legend=True):
    """Plot artificial missingness error across different number of clusters or weighths."""
    if kind == "Weight":
        data = ErrorAcrossWeights(distance_method, weights=list(np.arange(0, 55, 5)), ncl=ncl, n_runs=n_runs, tmt=7)
        title = "Weight Selection"
    elif kind == "Clusters":
        data = ErrorAcrossNumberOfClusters(distance_method, weight, n_runs=n_runs, tmt=7, n_clusters=list(np.arange(0, 31, 3)))
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


def ErrorAcrossNumberOfClusters(distance_method, weight, n_runs=5, tmt=7, n_clusters=[6, 9, 12, 15, 18, 21]):
    """Calculate missingness error across different number of clusters."""
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=tmt)
    X.index = np.arange(X.shape[0])
    md = X.copy()
    X = X.select_dtypes(include=['float64']).values
    errors = np.zeros((X.shape[0] * len(n_clusters) * n_runs, 9))
    for ii in range(n_runs):
        print("Run: ", ii)
        vals = FindIdxValues(md)
        md, nan_indices = IncorporateMissingValues(md, vals)
        data = md.select_dtypes(include=['float64']).T
        info = md.select_dtypes(include=['object'])
        missingness = (np.count_nonzero(np.isnan(data), axis=0) / data.shape[0] * 100).astype(float)
        baseline_errors = ComputeBaselineErrors(X, data.T, nan_indices)
        for jj, cluster in enumerate(n_clusters):
            print("#clusters: ", cluster)
            model = MassSpecClustering(info, cluster, weight, distance_method).fit(data, nRepeats=0)
            idx1 = X.shape[0] * ((ii * len(n_clusters)) + jj)
            idx2 = X.shape[0] * ((ii * len(n_clusters)) + jj + 1)
            errors[idx1:idx2, 0] = ii
            errors[idx1:idx2, 1] = md.index
            errors[idx1:idx2, 2] = missingness
            errors[idx1:idx2, 3] = cluster
            errors[idx1:idx2, 4] = ComputeModelError(X, data.T, nan_indices, model)
            errors[idx1:idx2, 5] = baseline_errors[0, :]  # Average
            errors[idx1:idx2, 6] = baseline_errors[1, :]  # Zero
            errors[idx1:idx2, 7] = baseline_errors[2, :]  # Minimum
            errors[idx1:idx2, 8] = baseline_errors[3, :]  # PCA

    df = pd.DataFrame(errors)
    df.columns = ["N_Run", "Peptide_Idx", "Missingness", "Clusters", "DDMC", "Average", "Zero", "Minimum", "PCA"]

    return df


def ErrorAcrossWeights(distance_method, weights, ncl=20, n_runs=5, tmt=7):
    """Calculate missingness error across different number of clusters."""
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=tmt)
    X.index = np.arange(X.shape[0])
    md = X.copy()
    X = X.select_dtypes(include=['float64']).values
    errors = np.zeros((X.shape[0] * len(weights) * n_runs, 9))
    for ii in range(n_runs):
        print("Run :", ii)
        vals = FindIdxValues(md)
        md, nan_indices = IncorporateMissingValues(md, vals)
        data = md.select_dtypes(include=['float64']).T
        info = md.select_dtypes(include=['object'])
        missingness = (np.count_nonzero(np.isnan(data), axis=0) / data.shape[0] * 100).astype(float)
        baseline_errors = ComputeBaselineErrors(X, data.T, nan_indices)
        for jj, w in enumerate(weights):
            print("Weight: ", w)
            model = MassSpecClustering(info, ncl, w, distance_method).fit(data, nRepeats=0)
            idx1 = X.shape[0] * ((ii * len(weights)) + jj)
            idx2 = X.shape[0] * ((ii * len(weights)) + jj + 1)
            errors[idx1:idx2, 0] = ii
            errors[idx1:idx2, 1] = md.index
            errors[idx1:idx2, 2] = missingness
            errors[idx1:idx2, 3] = model.SeqWeight
            errors[idx1:idx2, 4] = ComputeModelError(X, data.T, nan_indices, model)
            errors[idx1:idx2, 5] = baseline_errors[0, :]  # Average
            errors[idx1:idx2, 6] = baseline_errors[1, :]  # Zero
            errors[idx1:idx2, 7] = baseline_errors[2, :]  # Minimum
            errors[idx1:idx2, 8] = baseline_errors[3, :]  # PCA

    df = pd.DataFrame(errors)
    df.columns = ["N_Run", "Peptide_Idx", "Missingness", "Weight", "DDMC", "Average", "Zero", "Minimum", "PCA"]

    return df


def IncorporateMissingValues(X, vals):
    """Remove a random TMT experiment for each peptide. If a peptide already has the maximum amount of
    missingness allowed, don't remove."""
    d = X.select_dtypes(include=["float64"])
    tmt_idx = []
    for ii in range(d.shape[0]):
        tmt = random.sample(list(set(vals[vals[:, 0] == ii][:, -1])), 1)[0]
        a = vals[(vals[:, -1] == tmt) & (vals[:, 0] == ii)]
        tmt_idx.append((a[0, 0], a[:, 1]))
        X.iloc[a[0, 0], a[:, 1]] = np.nan
    return X, tmt_idx


def ComputeBaselineErrors(X, d, nan_indices, ncomp=5):
    """Compute error between baseline methods (i.e. average signal, minimum signal, zero, and PCA) and real value."""
    pc = PCA(d, ncomp=ncomp, missing="fill-em", method='nipals', standardize=False, demean=False, normalize=False)
    n = d.shape[0]
    errors = np.empty((4, n), dtype=float)
    for ii in range(n):
        idx = nan_indices[d.index[ii]]
        v = X[idx[0], idx[1] - 4]
        avE = [d.iloc[ii, :][~np.isnan(d.iloc[ii, :])].mean()] * v.size
        zeroE = [0.0] * v.size
        minE = [d.iloc[ii, :][~np.isnan(d.iloc[ii, :])].min()] * v.size
        pcaE = pc._adjusted_data[idx[0], idx[1] - 4]
        assert all(~np.isnan(v)) and all(~np.isnan(avE)) and all(~np.isnan(zeroE)) and all(~np.isnan(minE)) and all(~np.isnan(pcaE)), (v, avE, zeroE, minE, pcaE)
        errors[0, ii] = mean_squared_error(v, avE)
        errors[1, ii] = mean_squared_error(v, zeroE)
        errors[2, ii] = mean_squared_error(v, minE)
        errors[3, ii] = mean_squared_error(v, pcaE)
    return errors


def ComputeModelError(X, data, nan_indices, model):
    """Compute error between cluster center versus real value."""
    ncl = model.ncl
    labels = model.labels() - 1
    centers = model.transform().T
    n = data.shape[0]
    errors = np.empty(n, dtype=float)
    for ii in range(n):
        idx = nan_indices[data.index[ii]]
        v = X[idx[0], idx[1] - 4]
        c = centers[labels[ii], idx[1] - 4]
        assert all(~np.isnan(v)) and all(~np.isnan(c)), (v, c)
        errors[ii] = mean_squared_error(v, c)
    assert len(set(errors)) > 1, (centers, nan_indices[idx], v, c)
    return errors


def ComputeCenters(gmm, ncl):
    """Calculate cluster averages"""

    centers = np.zeros((ncl, gmm.distributions[0].d - 1))

    for ii, distClust in enumerate(gmm.distributions):
        for jj, dist in enumerate(distClust[:-1]):
            centers[ii, jj] = dist.parameters[0]

    return centers.T
