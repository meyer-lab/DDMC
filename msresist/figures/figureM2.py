"""
This creates Figure M1.
"""

import pickle
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
from ..pre_processing import filter_NaNpeptides, FindIdxValues
from ..binomial import position_weight_matrix, GenerateBinarySeqID, AAlist, BackgroundSeqs
from ..pam250 import MotifPam250Scores


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12.5, 12), (4, 3))
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    # d = X.select_dtypes(include=["float64"]).T

    # distance_method = "PAM250"

    # # Distribution of missingness per petide
    # plotMissingnessDensity(ax[0], d)

    # # Artificial missingness error across missingness percentages and corresponding wins
    # m_ = plotErrorAcrossMissingnessLevels(ax[1], X, [0, 0.35, 2], "PAM250", 5, 200, baseline=True)
    # plotWinsAcrossMissingnessLevels(ax[2:5], m_)

    # # Missingness error across number of clusters or different weights
    # plotErrorAcrossNumberOfClusters(ax[5], X, 0.45, "PAM250", np.arange(2, 21), 200)
    # plotErrorAcrossWeights(ax[6], X, [0, 0.1, 0.25, 0.5, 0.75, 1, 2], "PAM250", 10, 200)

    # # Run model
    X_f = filter_NaNpeptides(X, cut=0.1)
    d_f = X_f.select_dtypes(include=['float64']).T
    i_f = X_f.select_dtypes(include=['object'])
    pam_model = MassSpecClustering(i_f, ncl=15, SeqWeight=1, distance_method="PAM250").fit(d_f, "NA")

    import pickle
    with open('CPTACmodel_PAM250_W1_15CL', 'wb') as f:
        pickle.dump([pam_model], f)

    binom_model = MassSpecClustering(i_f, ncl=15, SeqWeight=10, distance_method="Binomial").fit(d_f, "NA")
    with open('CPTACmodel_BINOMIAL_W10_15CL', 'wb') as f:
        pickle.dump([binom_model], f)

    # centers = MSC.transform(d_f)
    # centers["Patient_ID"] = X.columns[4:]

    # # PCA of model
    # centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    # centers = TumorType(centers)
    # c = 2
    # plotPCA(
    #     ax[7:11], centers, c, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", hue_load="Cluster"
    # )

    # # Regress against survival
    # centers, y = TransformCPTACdataForRegression(MSC, d_f, list(X.columns[4:]))

    # centers_T = centers[~centers["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")
    # y_T = y[~y["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")

    # plsr = PLSRegression(n_components=2, scale=True)
    # plotR2YQ2Y(ax[11], plsr, centers_T, y_T, 1, 10)

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


def plotErrorAcrossMissingnessLevels(ax, x, weights, distance_method, ncl, max_n_iter=200):
    """Plot artificial missingness error."""
    errors = ErrorAcrossMissingnessLevels(x, weights, distance_method, ncl)
    m = pd.DataFrame(errors)
    m.columns = ["missingness%_group", "weights", "model_error", "baseline_error"]
    model_errors = m.iloc[:, :-1]
    sns.pointplot(x="missingness%_group", y="model_error", data=model_errors, hue="Weight", style="Fit", palette="muted", ax=ax)
    baseline_errors = m[["missingness%_group", "baseline_error"]]
    sns.pointplot(x="missingness%_group", y="baseline_error", data=baseline_errors, color="grey", ax=ax)
    ax.lines[-1].set_linestyle("--")
    return m


def ErrorAcrossMissingnessLevels(X, distance_method):
    """Incorporate different percentages of missing values in 'chunks' 8 observations and compute error 
    between the actual versus cluster center or imputed peptide average across patients. 
    Note that we start with the model fit to the entire data set."""
    #load model with all peptides >= 7 TMTs experiments
    models = []
    weights = ["Mix", "Seq", "Data"]
    if distance_method == "PAM250":
        with open('msresist/data/Results/CPTACmodel_PAM250_filteredTMT', 'rb') as m1: 
            models.append(pickle.load(m1)[0])
        with open('msresist/data/Results/CPTACmodel_PAM250_filteredTMT_seq', 'rb') as m2:
            models.append(pickle.load(m2)[0])
        with open('msresist/data/Results/CPTACmodel_PAM250_filteredTMT_data', 'rb') as m3:
            models.append(pickle.load(m3)[0])
    else:
        with open('msresist/data/Results/CPTACmodel_BINOMIAL_filteredTMT', 'rb') as m1:
            models.append(pickle.load(m1)[0])
        # with open('msresist/data/Results/CPTACmodel_BINOMIAL_filteredTMT_seq', 'rb') as m2:
        #     models.append(pickle.load(m2)[0])
        # with open('msresist/data/Results/CPTACmodel_BINOMIAL_filteredTMT_data', 'rb') as m3:
        #     models.append(pickle.load(m3)[0])

    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=7)
    md = X.copy()
    X = X.select_dtypes(include=['float64']).values
    errors = np.zeros((X.shape[0], 4))
    for _ in range(4):
        vals = FindIdxValues(md)
        md, nan_indices = IncorporateMissingValues(md, vals)
        d = md.select_dtypes(include=['float64'])
        errors[:, 0] = (np.count_nonzero(np.isnan(d), axis=1) / d.shape[1] * 100).astype(int)
        idxx = np.atleast_2d(np.arange(d.shape[0]))
        data = np.hstack((d, idxx.T))
        print(data.shape)
        for i, model in enumerate(models):
            fit = model.gmm_.fit(data, "NA")
            errors[:, 1] = weights[i]
            errors[:, 2] = ComputeModelError(X, fit, d, nan_indices)
            errors[:, 3] = ComputeBaselineError(X, d, nan_indices)

    return errors


def ComputeBaselineError(X, d, nan_indices):
    """Compute error between imputed average versus real value."""
    n = d.shape[0]
    errors = np.empty(n, dtype=float)
    for ii in range(n):
        idx = nan_indices[d.index[ii]]
        v = X[idx[0], idx[1] - 4]
        b = [d.iloc[ii, :][~np.isnan(d.iloc[ii, :])].mean()] * v.size
        errors[ii] = (mean_squared_error(v, b))
    return errors


def ComputeModelError(X, model, d, nan_indices):
    """Compute error between cluster center versus real value."""
    centers = model.transform()
    labels = model.labels_
    n = d.shape[0]
    errors = []
    for ii in range(n):
        idx = nan_indices[d.index[ii]]
        v = X[idx[0], idx[1] - 4]
        c = centers[labels[ii], idx[1] - 4]
        assert all(~np.isnan(v)) and all(~np.isnan(c)), (v, c)
        mse = mean_squared_error(v, c)
        errors.append(mse)
    return errors


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


def ErrorAcrossNumberOfClusters(X, weight, distance_method, clusters, max_n_iter):
    """Calculate missingness error across different number of clusters."""
    x, md, nan_indices = GenerateReferenceAndMissingnessDataSet(X)
    d = md.select_dtypes(include=['float64'])
    i = md.select_dtypes(include=['object'])

    # Pre-compute background
    seqs = [s.upper() for s in X["Sequence"]]
    if distance_method == "Binomial":
        # Background sequences
        bg = position_weight_matrix(BackgroundSeqs(md["Sequence"]))
        background = [np.array([bg[AA] for AA in AAlist]), GenerateBinarySeqID(seqs)]
    elif distance_method == "PAM250":
        # Compute all pairwise distances and generate seq vs seq to score dictionary
        background = MotifPam250Scores(seqs)

    model_res = np.zeros((len(clusters), 3))
    base_res = np.zeros((len(clusters), 2))
    for idx, cluster in enumerate(clusters):
        print(cluster)
        base_res[idx, 0] = int(cluster)
        model_res[idx, 0] = int(cluster)
        model = MassSpecClustering(
            i, cluster, SeqWeight=weight, distance_method=distance_method, max_n_iter=max_n_iter, background=background
        ).fit(d.T, "NA")
        if all(model.converge_):
            model_res[idx, 1] = 0
            model_res[idx, 2] = ComputeModelError(x, model, d, nan_indices)
        elif len(set(model.converge_)) == 2:
            model_res[idx, 1] = 1
            model_res[idx, 2] = ComputeModelError(x, model, d, nan_indices)
        else:
            model_res[idx, 1] = 2
            model_res[idx, 2] = np.nan

    base_res[:, 1] = [ComputeBaselineError(x, d, nan_indices)] * len(clusters)
    return model_res, base_res


def plotErrorAcrossNumberOfClusters(ax, X, weight, distance_method, clusters, max_n_iter):
    """Plot missingness error across different number of clusters."""
    m, b = ErrorAcrossNumberOfClusters(X, weight, distance_method, clusters, max_n_iter)
    m = pd.DataFrame(m)
    m.columns = ["n_clusters", "Fit", "Error"]
    sns.lineplot(x="n_clusters", y="Error", data=m, palette="muted", ax=ax)
    b = pd.DataFrame(b)
    b.columns = ["Clusters", "Error"]
    sns.lineplot(x="Clusters", y="Error", data=b, color="grey", ax=ax)
    ax.lines[-1].set_linestyle("--")


def ErrorAcrossWeights(X, weights, distance_method, ncl, max_n_iter):
    """Calculate missing error across different weights."""
    x, md, nan_indices = GenerateReferenceAndMissingnessDataSet(X)
    d = md.select_dtypes(include=['float64'])
    i = md.select_dtypes(include=['object'])

    # Pre-compute background
    seqs = [s.upper() for s in X["Sequence"]]
    if distance_method == "Binomial":
        # Background sequences
        bg = position_weight_matrix(BackgroundSeqs(md["Sequence"]))
        background = [np.array([bg[AA] for AA in AAlist]), GenerateBinarySeqID(seqs)]
    elif distance_method == "PAM250":
        # Compute all pairwise distances and generate seq vs seq to score dictionary
        background = MotifPam250Scores(seqs)

    model_res = np.zeros((len(weights), 3))
    base_res = np.zeros((len(weights), 2))
    for idx, w in enumerate(weights):
        print(w)
        base_res[idx, 0] = w
        model_res[idx, 0] = w
        model = MassSpecClustering(
            i, ncl, SeqWeight=w, distance_method=distance_method, max_n_iter=max_n_iter, background=background
        ).fit(d.T, "NA")
        if all(model.converge_):
            model_res[idx, 1] = 0
            model_res[idx, 2] = ComputeModelError(x, model, d, nan_indices)
        elif len(set(model.converge_)) == 2:
            model_res[idx, 1] = 1
            model_res[idx, 2] = ComputeModelError(x, model, d, nan_indices)
        else:
            model_res[idx, 1] = 2
            model_res[idx, 2] = np.nan

    base_res[:, 1] = [ComputeBaselineError(x, d, nan_indices)] * len(weights)
    return model_res, base_res


def plotErrorAcrossWeights(ax, X, weights, distance_method, ncl, max_n_iter):
    """Plot missingness error across different number of clusters."""
    m, b = ErrorAcrossWeights(X, weights, distance_method, ncl, max_n_iter)
    m = pd.DataFrame(m)
    m.columns = ["Weights", "Error"]
    sns.lineplot(x="Weights", y="Error", data=m, palette="muted", ax=ax)
    b = pd.DataFrame(b)
    b.columns = ["Weights", "Error"]
    sns.lineplot(x="Weights", y="Error", data=b, color="grey", ax=ax)
    ax.lines[-1].set_linestyle("--")


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
