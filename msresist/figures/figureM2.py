"""
This creates Figure M1.
"""

import pickle
import random
import numpy as np
from scipy.stats import gmean
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from .common import subplotLabel, getSetup
from .figure3 import plotR2YQ2Y, plotPCA
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides, FindIdxValues
from ..binomial import Binomial
from ..pam250 import PAM250
from ..expectation_maximization import EM_clustering


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((17, 12), (2, 3))

    # diagram explaining reconstruction process
    ax[0].axis("off")

    plotErrorAcrossMissingnessLevels(ax[1:4], "PAM250")
    plotErrorAcrossNumberOfClusters(ax[4], "PAM250")
    plotErrorAcrossWeights(ax[5], "PAM250")

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


def plotErrorAcrossNumberOfClusters(ax, distance_method):
    """Plot artificial missingness error across different number of clusters."""
    if distance_method == "PAM250":
        err = pd.read_csv("msresist/data/imputing_missingness/pam_c_5tmts.csv").iloc[:, 1:]
    else:
        err = pd.read_csv("msresist/data/imputing_missingness/binom_c_5tmts.csv").iloc[:, 1:]

    err.columns = ["Run", "pep_idx", "Miss", "n_clusters", "model_error", "base_error"]
    gm = pd.DataFrame(err.groupby(["n_clusters"]).model_error.apply(gmean)).reset_index()
    gm["model_error"] = np.log(gm["model_error"])
    gm["base_error"] = np.log(err.groupby(["n_clusters"]).base_error.apply(gmean).values)

    sns.regplot(x="n_clusters", y="model_error", data=gm, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5}, color="#001146", ax=ax)
    sns.regplot(x="n_clusters", y="base_error", data=gm, color="black", scatter=False, ax=ax)
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Imputation Error across Number of Clusters")


def plotErrorAcrossWeights(ax, distance_method):
    """Plot artificial missingness error across different number of clusters."""
    if distance_method == "PAM250":
        err = pd.read_csv("msresist/data/imputing_missingness/pam_w_5tmts.csv").iloc[:, 1:]
    else:
        err = pd.read_csv("msresist/data/imputing_missingness/binom_w_5tmts.csv").iloc[:, 1:]
    err.columns = ["Run", "pep_idx", "Miss", "Weight", "model_error", "base_error"]

    gm = pd.DataFrame(err.groupby(["Weight"]).model_error.apply(gmean)).reset_index()
    gm["model_error"] = np.log(gm["model_error"])
    gm["base_error"] = np.log(err.groupby(["Weight"]).base_error.apply(gmean).values)

    sns.regplot(x="Weight", y="model_error", data=gm, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5}, color="#001146", ax=ax)
    sns.regplot(x="Weight", y="base_error", data=gm, color="black", scatter=False, ax=ax)
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Imputation Error across Weights")


def plotErrorAcrossMissingnessLevels(ax, distance_method):
    """Plot artificial missingness error across verying missignenss."""
    if distance_method == "PAM250":
        err = pd.read_csv("msresist/data/imputing_missingness/pam_am_5tmts.csv").iloc[:, 1:]
    else:
        err = pd.read_csv("msresist/data/imputing_missingness/binom_am_5tmts.csv").iloc[:, 1:]

    err.columns = ["Run", "pep_idx", "Miss", "Weight", "model_error", "base_error"]
    gm = pd.DataFrame(err.groupby(["Weight", "Miss"]).model_error.apply(gmean)).reset_index()
    gm["model_error"] = np.log(gm["model_error"])
    gm["base_error"] = np.log(err.groupby(["Weight", "Miss"]).base_error.apply(gmean).values)

    data = gm[gm["Weight"] == 0.0]
    mix = gm[gm["Weight"] == 0.5]
    seq = gm[gm["Weight"] == 1.0]

    ylabel = "Mean Squared Error"
    sns.regplot(x="Miss", y="model_error", data=data, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5}, color="#acc2d9", ax=ax[0])
    sns.regplot(x="Miss", y="base_error", data=data, color="black", scatter=False, ax=ax[0])
    ax[0].set_title("Data only")
    ax[0].set_ylabel(ylabel)
    sns.regplot(x="Miss", y="model_error", data=mix, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5}, color="lightgreen", ax=ax[1])
    sns.regplot(x="Miss", y="base_error", data=mix, color="black", scatter=False, ax=ax[1])
    ax[1].set_title("Mix")
    ax[1].set_ylabel(ylabel)
    sns.regplot(x="Miss", y="model_error", data=seq, line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5}, color="#fdde6c", ax=ax[2])
    sns.regplot(x="Miss", y="base_error", data=seq, color="black", scatter=False, ax=ax[2])
    ax[2].set_title("Seq")
    ax[2].set_ylabel(ylabel)


def ErrorAcrossMissingnessLevels(distance_method):
    """Incorporate different percentages of missing values in 'chunks' 8 observations and compute error
    between the actual versus cluster center or imputed peptide average across patients.
    Note that we start with the model fit to the entire data set."""
    # load model with all peptides >= 7 TMTs experiments
    models = []
    weights = [0.5, 1.0, 0.0]
    if distance_method == "PAM250":
        with open('msresist/data/pickled_models/artificial_missingness/CPTACmodel_PAM250_filteredTMT', 'rb') as m1:
            models.append(pickle.load(m1)[0])
        with open('msresist/data/pickled_models/artificial_missingness/CPTACmodel_PAM250_filteredTMT_seq', 'rb') as m2:
            models.append(pickle.load(m2)[0])
        with open('msresist/data/pickled_models/artificial_missingness/CPTACmodel_PAM250_filteredTMT_data', 'rb') as m3:
            models.append(pickle.load(m3)[0])
    else:
        with open('msresist/data/pickled_models/artificial_missingness/CPTACmodel_BINOMIAL_filteredTMT', 'rb') as m1:
            models.append(pickle.load(m1)[0])
        with open('msresist/data/pickled_models/artificial_missingness/CPTACmodel_BINOMIAL_filteredTMT_seq', 'rb') as m2:
            models.append(pickle.load(m2)[0])
        with open('msresist/data/pickled_models/artificial_missingness/CPTACmodel_BINOMIAL_filteredTMT_data', 'rb') as m3:
            models.append(pickle.load(m3)[0])

    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=7)
    X.index = np.arange(X.shape[0])
    md = X.copy()
    X = X.select_dtypes(include=['float64']).values
    n_runs = 5
    errors = np.zeros((X.shape[0] * len(models) * n_runs, 6))
    for ii in range(n_runs):
        vals = FindIdxValues(md)
        md, nan_indices = IncorporateMissingValues(md, vals)
        data = md.select_dtypes(include=['float64']).T
        info = md.select_dtypes(include=['object'])
        missingness = (np.count_nonzero(np.isnan(data), axis=0) / data.shape[0] * 100).astype(float)
        for jj, model in enumerate(models):
            _, _, _, gmm = EM_clustering(data, info, model.ncl, gmmIn=model.gmm_)
            idx1 = X.shape[0] * ((ii * len(weights)) + jj)
            idx2 = X.shape[0] * ((ii * len(weights)) + jj + 1)
            errors[idx1:idx2, 0] = ii
            errors[idx1:idx2, 1] = md.index
            errors[idx1:idx2, 2] = missingness
            errors[idx1:idx2, 3] = weights[jj]
            errors[idx1:idx2, 4] = ComputeModelError(X, data.T, nan_indices, model.ncl, gmm, fit="gmm")
            errors[idx1:idx2, 5] = ComputeBaselineError(X, data.T, nan_indices)

    return errors


def ComputeBaselineError(X, d, nan_indices):
    """Compute error between imputed average versus real value."""
    n = d.shape[0]
    errors = np.empty(n, dtype=float)
    for ii in range(n):
        idx = nan_indices[d.index[ii]]
        v = X[idx[0], idx[1] - 4]
        b = [d.iloc[ii, :][~np.isnan(d.iloc[ii, :])].mean()] * v.size
        assert all(~np.isnan(v)) and all(~np.isnan(b)), (v, b)
        errors[ii] = mean_squared_error(v, b)
    return errors


def ComputeModelError(X, data, nan_indices, ncl, model, fit="gmm"):
    """Compute error between cluster center versus real value."""
    if fit == "gmm":
        d = np.array(data)
        idxx = np.atleast_2d(np.arange(d.shape[0]))
        d = np.hstack((d, idxx.T))
        labels = np.argmax(model.predict_proba(d), axis=1)
        centers = ComputeCenters(model, ncl).T
    else:
        labels = model.labels()
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


def ErrorAcrossNumberOfClusters(distance_method):
    """Calculate missingness error across different number of clusters."""
    if distance_method == "PAM250":
        weight = 1
    else:
        weight = 10

    n_clusters = [6, 9, 12, 15, 18, 21]
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=7)
    X.index = np.arange(X.shape[0])
    md = X.copy()
    X = X.select_dtypes(include=['float64']).values
    n_runs = 5
    errors = np.zeros((X.shape[0] * len(n_clusters) * n_runs, 6))
    for ii in range(n_runs):
        vals = FindIdxValues(md)
        md, nan_indices = IncorporateMissingValues(md, vals)
        data = md.select_dtypes(include=['float64']).T
        info = md.select_dtypes(include=['object'])
        missingness = (np.count_nonzero(np.isnan(data), axis=0) / data.shape[0] * 100).astype(float)
        for jj, cluster in enumerate(n_clusters):
            print(cluster)
            model = MassSpecClustering(info, cluster, weight, distance_method).fit(data, nRepeats=1)
            idx1 = X.shape[0] * ((ii * len(n_clusters)) + jj)
            idx2 = X.shape[0] * ((ii * len(n_clusters)) + jj + 1)
            errors[idx1:idx2, 0] = ii
            errors[idx1:idx2, 1] = md.index
            errors[idx1:idx2, 2] = missingness
            errors[idx1:idx2, 3] = cluster
            errors[idx1:idx2, 4] = ComputeModelError(X, data.T, nan_indices, model.ncl, model, fit="model")
            errors[idx1:idx2, 5] = ComputeBaselineError(X, data.T, nan_indices)

    return errors


def ErrorAcrossWeights(distance_method):
    """Calculate missingness error across different number of clusters."""
    if distance_method == "PAM250":
        with open('msresist/data/pickled_models/CPTACmodel_PAM250_filteredTMT', 'rb') as m:
            model = pickle.load(m)[0]
        weights = [0, 1, 3, 9, 27]
    else:
        with open('msresist/data/pickled_models/CPTACmodel_BINOMIAL_filteredTMT', 'rb') as m:
            model = pickle.load(m)[0]
        weights = [0, 5, 10, 20, 40]

    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=7)
    X.index = np.arange(X.shape[0])
    md = X.copy()
    X = X.select_dtypes(include=['float64']).values
    n_runs = 5
    errors = np.zeros((X.shape[0] * len(weights) * n_runs, 6))
    for ii in range(n_runs):
        vals = FindIdxValues(md)
        md, nan_indices = IncorporateMissingValues(md, vals)
        data = md.select_dtypes(include=['float64']).T
        info = md.select_dtypes(include=['object'])
        missingness = (np.count_nonzero(np.isnan(data), axis=0) / data.shape[0] * 100).astype(float)
        for jj, weight in enumerate(weights):
            print(weight)
            seqs = [s.upper() for s in info["Sequence"]]
            if distance_method == "PAM250":
                dist = PAM250(seqs, weight)
            elif distance_method == "Binomial":
                dist = Binomial(info["Sequence"], seqs, weight)
            _, _, _, gmm = EM_clustering(data, info, model.ncl, seqDist=dist, gmmIn=model.gmm_)
            idx1 = X.shape[0] * ((ii * len(weights)) + jj)
            idx2 = X.shape[0] * ((ii * len(weights)) + jj + 1)
            errors[idx1:idx2, 0] = ii
            errors[idx1:idx2, 1] = md.index
            errors[idx1:idx2, 2] = missingness
            errors[idx1:idx2, 3] = weight
            errors[idx1:idx2, 4] = ComputeModelError(X, data.T, nan_indices, model.ncl, gmm, fit="gmm")
            errors[idx1:idx2, 5] = ComputeBaselineError(X, data.T, nan_indices)

    return errors


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
