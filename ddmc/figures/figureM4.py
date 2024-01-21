"""
This creates Figure 4: Predictive performance of DDMC clusters using different weights
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ..clustering import DDMC
from .common import getSetup, HotColdBehavior, getDDMC_CPTAC
from ..logistic_regression import plotROC
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 3))

    X = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        tmt=2,
    )
    d = X.select_dtypes(include=[float]).T

    return f  # TODO: This code is broken.

    # Plot mean AUCs per model
    p = pd.read_csv("ddmc/data/Performance/preds_phenotypes_rs_15cl.csv").iloc[:, 1:]
    p = p.melt(
        id_vars=["Run", "Weight"],
        value_vars=d.columns[2:],
        value_name="p-signal",
        var_name="Phenotype",
    )
    out = d[
        (d["Weight"] == 0)
        | (d["Phenotype"] == "STK11") & (d["Weight"] == 1000)
        | (d["Phenotype"] == "EGFRm/ALKf") & (d["Weight"] == 250)
        | (d["Phenotype"] == "Infiltration") & (d["Weight"] == 250)
    ]
    out["Model"] = ["DDMC" if s != 0 else "GMM" for s in out["Weight"]]
    sns.barplot(data=out, x="Phenotype", y="p-signal", hue="Model", ci=68, ax=ax[0])
    ax[0].legend(prop={"size": 5}, loc=0)

    # Fit Data, Mix, and Seq Models
    dataM = getDDMC_CPTAC(n_components=30, SeqWeight=0.0)
    mixM = getDDMC_CPTAC(n_components=30, SeqWeight=250.0)
    seqM = getDDMC_CPTAC(n_components=30, SeqWeight=1.0e6)
    models = [dataM, mixM, seqM]

    # Center to peptide distance
    barplot_PeptideToClusterDistances(models, ax[1], n=2000)

    # Position Enrichment
    boxplot_TotalPositionEnrichment(models, ax[2])

    return f


def calculate_AUCs_phenotypes(ax, X: pd.DataFrame, nRuns=3, n_components=35):
    """Plot mean AUCs per phenotype across weights."""
    # Signaling
    d = X.select_dtypes(include=[float]).T
    i = X["Sequence"]

    # Genotype data
    mutations = pd.read_csv("ddmc/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[
        ["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]
    ]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)

    # LASSO
    lr = LogisticRegressionCV(
        cv=3,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )

    weights = [0, 50, 100, 250, 500, 750, 1000, 1000000]
    run, ws, stk, ea, hcb = [], [], [], [], []
    for w in weights:
        for r in range(nRuns):
            run.append(r)
            ws.append(w)
            model = DDMC(
                i, n_components=n_components, SeqWeight=w, distance_method="Binomial"
            ).fit(d)

            # Find and scale centers
            centers_gen, centers_hcb = TransformCenters(model, X)

            # STK11
            stk.append(
                plotROC(
                    ax,
                    lr,
                    centers_gen.values,
                    y["STK11.mutation.status"].values,
                    cv_folds=3,
                    return_mAUC=True,
                    kfold="Repeated",
                )
            )

            # EGFRm/ALKf
            y_EA = merge_binary_vectors(y.copy(), "EGFR.mutation.status", "ALK.fusion")
            ea.append(
                plotROC(
                    ax,
                    lr,
                    centers_gen.values,
                    y_EA,
                    cv_folds=3,
                    return_mAUC=True,
                    kfold="Repeated",
                )
            )

            # Hot-Cold behavior
            y_hcb, centers_hcb = HotColdBehavior(centers_hcb)
            hcb.append(
                plotROC(
                    ax,
                    lr,
                    centers_hcb.values,
                    y_hcb,
                    cv_folds=3,
                    return_mAUC=True,
                    kfold="Repeated",
                )
            )

    out = pd.DataFrame()
    out["Run"] = run
    out["Weight"] = ws
    out["STK11"] = stk
    out["EGFRm/ALKf"] = ea
    out["Infiltration"] = hcb

    return out


def barplot_PeptideToClusterDistances(models, ax, n=3000):
    """Compute and plot p-signal-to-center and motif to cluster distance for n peptides across weights."""
    # Import signaling data, select random peptides, and find cluster assignments
    X = pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    random_peptides = np.random.choice(
        list(np.arange(len(models[0].labels()))), n, replace=False
    )
    X["labels0"] = models[0].labels()
    X["labels500"] = models[1].labels()
    X["labels1M"] = models[2].labels()
    X = X.iloc[random_peptides, :]
    labels = X.loc[:, ["labels0", "labels500", "labels1M"]].values.T
    d = X.select_dtypes(include=[float]).values

    psDist = np.zeros((3, d.shape[0]))
    for ii, model in enumerate(models):
        for jj in range(d.shape[0]):
            # Data distance
            center = model.transform()[:, labels[ii, jj] - 1]
            idx_values = np.argwhere(~np.isnan(d[jj, :]))
            psDist[ii, jj] = mean_squared_error(d[jj, idx_values], center[idx_values])

    psDist = pd.DataFrame(psDist).T
    psDist.columns = ["p-Abundance", "Mix", "Sequence"]
    ps_data = pd.melt(
        psDist,
        value_vars=["p-Abundance", "Mix", "Sequence"],
        var_name="Model",
        value_name="p-signal MSE",
    )
    sns.barplot(data=ps_data, x="Model", y="p-signal MSE", ci=None, ax=ax)
    ax.set_title("Peptide-to-Cluster signal MSE")


def boxplot_TotalPositionEnrichment(models, ax):
    """Position enrichment of cluster PSSMs"""
    enr = np.zeros((3, models[0].n_components), dtype=float)
    for ii, model in enumerate(models):
        pssms = model.pssms()[0]
        for jj in range(len(pssms)):
            enr[ii, jj] = np.sum(pssms[jj].sum().drop(5))

    enr = pd.DataFrame(enr)
    enr.columns = np.arange(models[0].n_components) + 1
    enr.insert(0, "Model", ["p-Abundance", "Mix", "Sequence"])
    dd = pd.melt(
        frame=enr,
        id_vars="Model",
        value_vars=enr.columns[1:],
        var_name="Cluster",
        value_name="Total information (bits)",
    )
    sns.stripplot(data=dd, x="Model", y="Total information (bits)", ax=ax)
    sns.boxplot(data=dd, x="Model", y="Total information (bits)", ax=ax, fliersize=0)
    ax.set_title("Cumulative PSSM Enrichment")


def merge_binary_vectors(y, mutant1, mutant2):
    """Merge binary mutation status vectors to identify all patients having one of the two mutations"""
    y1 = y[mutant1]
    y2 = y[mutant2]
    y_ = np.zeros(y.shape[0])
    for binary in [y1, y2]:
        indices = [i for i, x in enumerate(binary) if x == 1]
        y_[indices] = 1
    return pd.Series(y_)


def find_patients_with_NATandTumor(X: pd.DataFrame, label, conc=False) -> pd.DataFrame:
    """Reshape data to display patients as rows and samples (Tumor and NAT per cluster) as columns.
    Note that to do so, samples that don't have their tumor/NAT counterpart are dropped.
    """
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
        xN.columns = [str(i) + "_Normal" for i in xN.columns]
        X.columns = [str(i) + "_Tumor" for i in X.columns]
        X = pd.concat([X, xN], axis=1)
    return X


def TransformCenters(model: DDMC, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For a given model, find centers and transform for regression."""
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(
        centers.iloc[:, :]
    )
    centers = centers.T
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers1 = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    centers2 = (
        centers.loc[~centers["Patient_ID"].str.endswith(".N"), :]
        .sort_values(by="Patient_ID")
        .set_index("Patient_ID")
    )
    return centers1, centers2
