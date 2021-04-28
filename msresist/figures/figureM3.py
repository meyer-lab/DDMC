"""
This creates Figure 3: Predictive performance of DDMC clusters using different weights
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from .common import subplotLabel, getSetup
from ..logistic_regression import plotROC
from ..pre_processing import filter_NaNpeptides
from .figure2 import plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Signaling

    # Plot mean AUCs per model
    models = plotAUCs(ax[0], return_models=True)

    # Center to peptide distance
    barplot_PeptideToClusterDistances(models, ax[1], n=2000)

    # Position Enrichment
    boxplot_PositionEnrichment(models, ax[2])

    # Motifs
    plot_SpecificPeptide(models, ax[3:8], peptide="MGRKEsEEELE", yaxis=[0, 7])

    return f


def plotAUCs(ax, return_models=False):
    """Plot mean AUCs per phenotype across weights."""
    # Signaling
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    # Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)

    # LASSO
    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    folds = 5
    weights = [0, 15, 20, 40, 50]
    path = 'msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W'
    aucs = np.zeros((3, 5), dtype=float)
    models = []
    for ii, w in enumerate(weights):
        with open(path + str(w) + '_TMT2', 'rb') as m:
            model = pickle.load(m)[0]

        if return_models and w in [0, 20, 50]:
            models.append(model)

        # Find and scale centers
        centers_gen, centers_hcb = TransformCenters(model, X)

        # STK11
        aucs[0, ii] = plotROC(ax, lr, centers_gen.values, y["STK11.mutation.status"], cv_folds=folds, return_mAUC=True)

        # EGFRm/ALKf
        y_EA = merge_binary_vectors(y.copy(), "EGFR.mutation.status", "ALK.fusion")
        aucs[1, ii] = plotROC(ax, lr, centers_gen.values, y_EA, cv_folds=folds, return_mAUC=True)

        # Hot-Cold behavior
        y_hcb, centers_hcb = HotColdBehavior(centers_hcb)
        aucs[2, ii] = plotROC(ax, lr, centers_hcb.values, y_hcb, cv_folds=folds, return_mAUC=True)

    res = pd.DataFrame(aucs)
    res.columns = [str(w) for w in weights]
    res["Phenotype"] = ["STK11m", "EGFRm/ALKf", "Infiltration"]
    data = pd.melt(frame=res, id_vars="Phenotype", value_vars=res.columns[:-1], var_name="Weight", value_name="mean AUC")
    sns.lineplot(data=data, x="Weight", y="mean AUC", hue="Phenotype", ax=ax)
    ax.set_title("Predictive performance by Weight")

    if return_models:
        return models


def barplot_PeptideToClusterDistances(models, ax, n=3000):
    """Compute and plot p-signal-to-center and motif to cluster distance for n peptides across weights."""
    # Import signaling data, select random peptides, and find cluster assignments
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    random_peptides = np.random.choice(list(np.arange(len(models[0].labels()))), n, replace=False)
    X["labels0"] = models[0].labels()
    X["labels20"] = models[1].labels()
    X["labels50"] = models[2].labels()
    X = X.iloc[random_peptides, :]
    labels = X.loc[:, ["labels0", "labels20", "labels50"]].values.T
    d = X.select_dtypes(include=[float]).values

    psDist = np.zeros((3, d.shape[0]))
    for ii, model in enumerate(models):
        for jj in range(d.shape[0]):
            # Data distance
            center = model.transform()[:, labels[ii, jj] - 1]
            idx_values = np.argwhere(~np.isnan(d[jj, :]))
            psDist[ii, jj] = mean_squared_error(d[jj, idx_values], center[idx_values])

    psDist = pd.DataFrame(psDist).T
    psDist.columns = ["Data", "Mix", "Sequence"]
    ps_data = pd.melt(psDist, value_vars=["Data", "Mix", "Sequence"], var_name="Model", value_name="p-signal MSE")
    sns.barplot(data=ps_data, x="Model", y="p-signal MSE", ci=None, ax=ax)
    ax.set_title("Peptide-to-Cluster signal MSE")


def boxplot_PositionEnrichment(models, ax):
    """Position enrichment of cluster PSSMs"""
    enr = np.zeros((3, 24), dtype=float)
    for ii, model in enumerate(models):
        pssms = model.pssms()
        for jj in range(models[0].ncl):
            enr[ii, jj] = np.sum(pssms[jj].sum().drop(5))

    enr = pd.DataFrame(enr)
    enr.columns = np.arange(models[0].ncl) + 1
    enr.insert(0, "Model", ["Data", "Mix", "Sequence"])
    dd = pd.melt(frame=enr, id_vars="Model", value_vars=enr.columns[1:], var_name="Cluster", value_name="Total information (bits)")
    sns.stripplot(data=dd, x="Model", y="Total information (bits)", ax=ax)
    sns.boxplot(data=dd, x="Model", y="Total information (bits)", ax=ax)
    ax.set_title("Cumulative PSSM Enrichment")


def plot_SpecificPeptide(models, ax, peptide="MGRKEsEEELE", yaxis=False):
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    seqs = [s.upper() for s in X["Sequence"].values]
    X["labels0"] = models[0].labels()
    X["labels20"] = models[1].labels()
    X["labels50"] = models[2].labels()
    if not peptide:
        peptide = random.sample(list(np.arange(len(models[0].labels()))), 1)
        X = pd.DataFrame(X.iloc[peptide, :])
    else:
        X = pd.DataFrame(X.set_index("Sequence").loc[peptide, :]).T.reset_index()
        X.columns = ["Sequence"] + list(X.columns)[1:]

    labels = np.squeeze(X.loc[:, ["labels0", "labels20", "labels50"]].values.T)

    pds_names = [str(s).split("('")[1].split("')")[0].replace("'", "").replace(",", ";") for s in zip(list(X["Gene"]), list(X["Sequence"]))]
    model_names = ["Data", "Mix", "Sequence"]

    d = X.iloc[:, 4:-3].values.astype(float)
    mat = np.zeros((3, 1))
    for ii, model in enumerate(models):
        # Distance Calculation
        center = model.transform()[:, labels[ii] - 1]
        idx_values = np.argwhere(~np.isnan(d))
        idx_values = [w[1] for w in idx_values]
        mat[ii, 0] = mean_squared_error(d[0][idx_values], center[idx_values])

    # Plot peptide-to-cluster p-signal MSE
    mat = pd.DataFrame(mat)
    mat.columns = pds_names
    mat.insert(0, "Model", model_names)
    data = pd.melt(frame=mat, id_vars="Model", value_vars=mat.columns[1:], var_name="Peptide", value_name="p-signal MSE")
    sns.barplot(data=data, x="Peptide", y="p-signal MSE", hue="Model", ax=ax[0])
    ax[0].set_title("Peptide-to-Cluster signal MSE")
    ax[0].legend(prop={'size': 8}, loc='lower right')

    # Plot Position Enrichment
    enr = np.zeros((3, 1), dtype=float)
    for ii, model in enumerate(models):
        pssms = model.pssms()
        enr[ii, 0] = np.sum(pssms[labels[ii]].sum().drop(5))

    enr = pd.DataFrame(enr).T
    enr.columns = ["Data", "Mix", "Sequence"]
    dd = pd.melt(frame=enr, value_vars=enr.columns, var_name="Model", value_name="Position Enrichment")
    sns.barplot(data=dd, x="Model", y="Position Enrichment", ax=ax[1])
    ax[1].set_title("Cluster PSSM Enrichment")

    # Plot Motifs
    pssms = [model.pssms()[labels[ii]] for ii, model in enumerate(models)]
    plotMotifs(pssms, axes=ax[2:5], titles=["Data", "Mix", "Sequence"], yaxis=yaxis)


def merge_binary_vectors(y, mutant1, mutant2):
    """Merge binary mutation status vectors to identify all patients having one of the two mutations"""
    y1 = y[mutant1]
    y2 = y[mutant2]
    y_ = np.zeros(y.shape[0])
    for binary in [y1, y2]:
        indices = [i for i, x in enumerate(binary) if x == 1]
        y_[indices] = 1
    return pd.Series(y_)


def find_patients_with_NATandTumor(X, label, conc=False):
    """Reshape data to display patients as rows and samples (Tumor and NAT per cluster) as columns.
    Note that to do so, samples that don't have their tumor/NAT counterpart are dropped."""
    xT = X[~X[label].str.endswith(".N")].sort_values(by=label)
    xN = X[X[label].str.endswith(".N")].sort_values(by=label)
    l1 = list(xT[label])
    l2 = [s.split(".N")[0] for s in xN[label]]
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    X = xT.set_index(label).drop(dif)
    assert all(X.index.values == np.array(l2)), "Samples don't match"

    if conc:
        xN = xN.set_index(label)
        xN.index = l2
        xN.columns = [str(i) + "_N" for i in xN.columns]
        X.columns = [str(i) + "_T" for i in X.columns]
        X = pd.concat([X, xN], axis=1)
    return X


def TransformCenters(model, X):
    """For a given model, find centers and transform for regression."""
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers1 = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    centers2 = centers.loc[~centers["Patient_ID"].str.endswith(".N"), :].sort_values(by="Patient_ID").set_index("Patient_ID")
    return centers1, centers2


def HotColdBehavior(centers):
    # Import Cold-Hot Tumor data
    y = pd.read_csv("msresist/data/MS/CPTAC/Hot_Cold.csv").dropna(axis=1).sort_values(by="Sample ID")
    y = y.loc[~y["Sample ID"].str.endswith(".N"), :].set_index("Sample ID")
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)

    # Transform to binary
    y = y.replace("Cold-tumor enriched", 0)
    y = y.replace("Hot-tumor enriched", 1)
    y = np.squeeze(y)

    # Remove NAT-enriched samples
    centers = centers.drop(y[y == "NAT enriched"].index)
    y = y.drop(y[y == "NAT enriched"].index).astype(int)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    return y, centers
