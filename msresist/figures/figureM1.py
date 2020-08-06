"""
This creates Figure M1.
"""

import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from .common import subplotLabel, getSetup
from .figure3 import plotR2YQ2Y, plotPCA
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12.5, 12), (4, 3))
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    d = X.select_dtypes(include=["float64"]).T

    distance_method = "PAM250"
    NaNfilter = 0.1

    # Distribution of missingness per petide
    plotMissingnessDensity(ax[0], d)

    # Artificial missingness error across missingness percentages and corresponding wins
    m_ = plotErrorAcrossMissingnessLevels(ax[1], X, NaNfilter, [0, 0.35, 2], "PAM250", 5, 200, baseline=True)
    plotWinsAcrossMissingnessLevels(ax[2:5], m_, [0, 0.35, 2])

    # Missingness error across number of clusters or different weights
    plotErrorAcrossNumberOfClusters(ax[5], X, NaNfilter, 0.45, "PAM250", np.arange(2, 21), 200)
    plotErrorAcrossWeights(ax[6], X, NaNfilter, [0, 0.1, 0.25, 0.5, 0.75, 1, 2], "PAM250", 10, 200)

    # Run model
    X_f = filter_NaNpeptides(X, cut=0.1)
    d_f = X_f.select_dtypes(include=['float64']).T
    i_f = X_f.select_dtypes(include=['object'])
    distance_method = "PAM250"
    ncl = 19
    SeqWeight = 0.75
    MSC = MassSpecClustering(
        i_f, ncl, SeqWeight=SeqWeight, distance_method=distance_method, n_runs=1
    ).fit(d_f, "NA")
    centers = MSC.transform(d_f)
    centers["Patient_ID"] = X.columns[4:]

    # PCA of model
    centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    centers = TumorType(centers)
    c = 2
    plotPCA(
        ax[7:11], centers, c, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", hue_load="Cluster"
    )

    # Regress against survival
    centers, y = TransformCPTACdataForRegression(MSC, d_f, list(X.columns[4:]))

    centers_T = centers[~centers["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")
    y_T = y[~y["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")

    plsr = PLSRegression(n_components=2, scale=True)
    plotR2YQ2Y(ax[11], plsr, centers_T, y_T, 1, 10)

    # Add subplot labels
    subplotLabel(ax)

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


def plotErrorAcrossMissingnessLevels(ax, x, NaNfilter, weights, distance_method, ncl, max_n_iter=200, baseline=False):
    """Plot artificial missingness error."""
    m, b = ErrorAcrossMissingnessLevels(x, NaNfilter, weights, distance_method, ncl, max_n_iter=max_n_iter)
    m = pd.DataFrame(m)
    m.columns = ["Missing%", "Weight", "SeqWins", "DataWins", "BothWin", "MixWin", "Error"]
    sns.lineplot(x="Missing%", y="Error", data=m, hue="Weight", palette="muted", ax=ax)
    if baseline:
        b = pd.DataFrame(b)
        b.columns = ["Missing%", "Error"]
        sns.lineplot(x="Missing%", y="Error", data=b, color="grey", ax=ax)
        ax.lines[-1].set_linestyle("--")
    return m


def plotWinsAcrossMissingnessLevels(ax, X, weights):
    """Plot all wins across missingness percentages per weight generated in PlotArtificialMissingnessError."""
    for r in range(X.shape[0]):
        X.iloc[r, 2:6] = X.iloc[r, 2:6].div(X.iloc[r, 2:6].sum())
    x = pd.melt(
        X, id_vars=['Weight', 'Missing%', 'Error'], value_vars=['SeqWins', 'DataWins', 'BothWin', 'MixWin'], 
        var_name="Winner", value_name='Wins'
    )
    for i in range(len(ax)):
        d = x[x["Weight"] == weights[i]]
        sns.barplot(x="Missing%", y="Wins", hue="Winner", data=d, ax=ax[i])
        ax[i].set_title("Weight: " + str(weights[i]))
        ax[i].get_legend().remove()
    ax[-1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2)


def ErrorAcrossMissingnessLevels(X, NaNfilter, weights, distance_method, ncl, max_n_iter):
    """Incorporate different percentages of missing values in 'chunks' 8 observations and compute error between the actual
    versus cluster average value. Note that the wins for all fitted models are returned to be used in PlotAMwins."""
    sc = [0, 2, 4, 6, 8]
    nan_per = [0, 10, 25, 50, 75]
    x, md, nan_indices = GenerateReferenceAndMissingnessDataSet(X, NaNfilter)
    assert md.equals(x) == False, "NaNs were not added."
    groups = MissingnessGroups(md, nan_per)
    md["MissingnessGroups"] = groups

    # Compute Error for each missingness group and each weight
    model_res = np.zeros(((len(nan_per)) * len(weights), 7))
    base_res = np.zeros((len(nan_per), 2))
    for ii in range(len(nan_per)):
        data = md[md["MissingnessGroups"] == ii].iloc[:, :-1]
        assert data.empty == False, "Empty missingness group."
        d = data.select_dtypes(include=['float64'])
        i = data.select_dtypes(include=['object'])
        base_res[ii, 1] = ComputeBaselineError(x, d, nan_indices)
        base_res[ii, 0] = nan_per[ii]

        for jj in range(len(weights)):
            model = MassSpecClustering(
                i, ncl, SeqWeight=weights[jj], distance_method=distance_method, max_n_iter=max_n_iter
            ).fit(d.T, "NA")
            model_res[ii + sc[ii] + jj, 0] = int(nan_per[ii])
            model_res[ii + sc[ii] + jj, 1] = weights[jj]
            model_res[ii + sc[ii] + jj, 2] = model.wins_[0]
            model_res[ii + sc[ii] + jj, 3] = model.wins_[1]
            model_res[ii + sc[ii] + jj, 4] = model.wins_[2]
            model_res[ii + sc[ii] + jj, 5] = model.wins_[3]
            model_res[ii + sc[ii] + jj, 6] = ComputeModelError(x, model, d, nan_indices)

    return model_res, base_res


def GenerateReferenceAndMissingnessDataSet(X, NaNfilter):
    """Generate data set with the incorporated missing values"""
    x = filter_NaNpeptides(X, cut=NaNfilter)
    x.index = np.arange(x.shape[0])
    md = x.copy()
    x = x.iloc[:, 4:].values
    vals = FindIdxValues(md)
    md, nan_indices = IncorporateMissingValues(md, vals)
    return x, md, nan_indices


def ComputeBaselineError(X, d, nan_indices):
    """Compute error between imputed average versus real value."""
    n = d.shape[0]
    errors = np.empty(n, dtype=float)
    for ii in range(n):
        idx = nan_indices[d.index[ii]]
        v = X[idx[0], idx[1] - 4]
        b = [d.iloc[ii, :][~np.isnan(d.iloc[ii, :])].mean()] * v.size
        errors[ii] = (mean_squared_error(v, b))
    if np.isnan(np.mean(errors)):
        display(pd.DataFrame(errors).T)
        raise SystemExit
    return np.mean(errors)


def ComputeModelError(X, model, d, nan_indices):
    """Compute error between cluster center versus real value."""
    centers = model.transform(d.T).T.values
    labels = model.labels_
    n = d.shape[0]
    errors = np.empty(n, dtype=float)
    for ii in range(n):
        idx = nan_indices[d.index[ii]]
        v = X[idx[0], idx[1] - 4]
        c = centers[labels[ii], idx[1] - 4]
        mse = mean_squared_error(v, c)
        errors[ii] = mse
    if np.isnan(np.mean(errors)):
        display(pd.DataFrame(errors).T)
        raise SystemExit
    return np.mean(errors)


def MissingnessGroups(X, l):
    """Assign each peptide to the closest missingness group."""
    d = X.select_dtypes(include=["float64"])
    pept_NaN_per = (np.count_nonzero(np.isnan(d), axis=1) / d.shape[1] * 100).astype(int)
    l_index = []
    for per in pept_NaN_per:
        l_index.append(l.index(min(l, key=lambda group: abs(group - per))))
    assert max(l_index) + 1 == len(l), "Not enough missingness present in input data set"
    return l_index


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


def FindIdxValues(X):
    """Find the patient indices corresponding to all non-missing values grouped in TMT experiments. Only
    value variables should be passed."""
    data = X.select_dtypes(include=["float64"])
    idx = np.argwhere(~np.isnan(data.values))
    idx[:, 1] += 4 #add ID variable columns
    StoE = pd.read_csv("msresist/data/MS/CPTAC/IDtoExperiment.csv")
    assert all(StoE.iloc[:, 0] == data.columns), "Sample labels don't match."
    StoE = StoE.iloc[:, 1].values
    tmt = [[StoE[idx[ii][1] - 4]] for ii in range(idx.shape[0])]
    return np.append(idx, tmt, axis=1)


def ErrorAcrossNumberOfClusters(X, NaNfilter, weight, distance_method, clusters, max_n_iter):
    """Calculate missingness error across different number of clusters."""
    x, md, nan_indices = GenerateReferenceAndMissingnessDataSet(X, NaNfilter)
    d = md.select_dtypes(include=['float64'])
    i = md.select_dtypes(include=['object'])

    res = np.zeros((len(clusters), 2))
    for idx, cluster in enumerate(clusters):
        model = MassSpecClustering(
                i, cluster, SeqWeight=weight, distance_method=distance_method, max_n_iter=max_n_iter
            ).fit(d.T, "NA")
        res[idx, 0] = int(cluster)
        res[idx, 1] = ComputeModelError(x, model, d, nan_indices)

    return res


def plotErrorAcrossNumberOfClusters(ax, X, NaNfilter, weight, distance_method, clusters, max_n_iter):
    """Plot missingness error across different number of clusters."""
    res = ErrorAcrossNumberOfClusters(X, NaNfilter, weight, distance_method, clusters, max_n_iter)
    res = pd.DataFrame(res)
    res.columns = ["n_clusters", "Error"]
    sns.lineplot(x="n_clusters", y="Error", data=res, palette="muted", ax=ax)


def ErrorAcrossWeights(X, NaNfilter, weights, distance_method, ncl, max_n_iter):
    """Calculate missing error across different weights."""
    x, md, nan_indices = GenerateReferenceAndMissingnessDataSet(X, NaNfilter)
    d = md.select_dtypes(include=['float64'])
    i = md.select_dtypes(include=['object'])

    res = np.zeros((len(weights), 2))
    for idx, w in enumerate(weights):
        model = MassSpecClustering(
                i, ncl, SeqWeight=w, distance_method=distance_method, max_n_iter=max_n_iter
            ).fit(d.T, "NA")
        res[idx, 0] = w
        res[idx, 1] = ComputeModelError(x, model, d, nan_indices)
    return res


def plotErrorAcrossWeights(ax, X, NaNfilter, weights, distance_method, ncl, max_n_iter):
    """Plot missingness error across different number of clusters."""
    res = ErrorAcrossWeights(X, NaNfilter, weights, distance_method, ncl, max_n_iter)
    res = pd.DataFrame(res)
    res.columns = ["Weights", "Error"]
    sns.lineplot(x="Weights", y="Error", data=res, palette="muted", ax=ax)


def TumorType(centers):
    """Add Normal vs Tumor column."""
    tumortype = []
    for i in range(centers.shape[0]):
        if ".N" in centers["Patient_ID"][i]:
            tumortype.append("Normal")
        else:
            tumortype.append("Tumor")
    centers["Type"] = tumortype
    return centers


def TransformCPTACdataForRegression(model, d, patient_IDs):
    """Match patient IDs to clinical outcomes for regression and return phosphoproteomic and clinical data sets."""
    centers = model.transform(d)
    centers["Patient_ID"] = patient_IDs

    # Import Vital Status after 12 months and transform to binary
    cf = pd.read_csv("msresist/data/MS/CPTAC/CPTACLUAD_VitalStatus.csv")
    cf = cf.replace("Living", 0)
    cf = cf.replace("Deceased", 1)

    # Import dictionary with sample IDs to map patients in both data sets
    IDict = pd.read_csv("msresist/data/MS/CPTAC/IDs.csv", header=0)
    IDict_ = dict(zip(IDict.iloc[:, 0], IDict.iloc[:, 1]))
    cf = SwapPatientIDs(cf, IDict_)
    cf = AddTumorPerPatient(cf)

    centers = centers.set_index("Patient_ID").sort_values(by="Patient_ID")
    y = cf.sort_values(by="Patient_ID").set_index("Patient_ID")

    # Check differences in patients present in both data sets
    diff = list(set(centers.index) - set(y.index))
    centers = centers.drop(diff)
    assert len(diff) < 10, "missing many samples"

    # Drop patients for which there is no vital status and assert all patient IDs match
    nans = y[np.isnan(y["vital_status_12months"])].index
    y = y.dropna().reset_index()
    centers = centers.drop(nans).reset_index()
    assert all(centers.index == y.index), "samples not matching."
    return centers, y


def SwapPatientIDs(cf, IDict_):
    """Change patient IDs from Case ID to Broad ID."""
    ids = []
    for i in range(cf.shape[0]):
        ids.append(IDict_[cf.iloc[i, 0]])
    cf["Patient_ID"] = ids
    return cf


def AddTumorPerPatient(cf):
    """Add Tumor row per patient in vital status data."""
    for i in range(cf.shape[0]):
        id_ = cf.iloc[i, 0]
        if ".N" in id_:
            iD = id_.split(".N")[0]
            cf.loc[-1] = [iD, cf.iloc[i, 1]]
            cf.index = cf.index + 1
        else:
            continue
    cf.index = cf.index + 1
    return cf.sort_index()
